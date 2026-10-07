"""Provider-agnostic chat model factory.

PROVIDER   openai | anthropic | litellm | local
MODEL      model name as the provider expects it (e.g. gpt-4o-mini, claude-sonnet-4-5, ollama model tag)
API_KEY    key for the provider (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY / LITELLM_API_KEY)
BASE_URL   optional endpoint override (falls back to OPENAI_BASE_URL)

`local` and `litellm` talk to any OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp, LiteLLM proxy).
"""
import json, os, time
from dataclasses import dataclass, asdict

PROVIDERS = ['openai', 'anthropic', 'litellm', 'local']
DEFAULT_BASE_URLS = {
    'openai': 'https://api.openai.com/v1',
    'anthropic': 'https://api.anthropic.com',
    'litellm': 'http://localhost:4000',
    'local': 'http://localhost:11434/v1',
}
KEY_ENV = {'openai': 'OPENAI_API_KEY', 'anthropic': 'ANTHROPIC_API_KEY', 'litellm': 'LITELLM_API_KEY', 'local': None}


@dataclass
class LLMConfig:
    provider: str = 'openai'
    model: str = ''
    api_key: str = ''
    base_url: str = ''
    temperature: float | None = 0.7
    max_tokens: int = 2000
    timeout: float = 180
    context_chars: int = 24000       # most text sent to the model in one reporter call; longer material is condensed in pieces

    @classmethod
    def from_env(cls):
        provider = (os.getenv('PROVIDER') or 'openai').strip().lower()
        key_env = KEY_ENV.get(provider)
        temp = os.getenv('TEMPERATURE', '0.7').strip()
        return cls(
            provider=provider,
            model=os.getenv('MODEL', '').strip(),
            api_key=(os.getenv('API_KEY') or (os.getenv(key_env) if key_env else '') or '').strip(),
            base_url=(os.getenv('BASE_URL') or (os.getenv('OPENAI_BASE_URL') if provider == 'openai' else '') or '').strip(),
            temperature=None if temp.lower() in ('', 'none', 'default') else float(temp),
            max_tokens=int(os.getenv('MAX_TOKENS', '2000')),
            timeout=float(os.getenv('LLM_TIMEOUT', '180')),
            context_chars=int(os.getenv('CONTEXT_CHARS', '24000')),
        )

    def validate(self):
        if self.provider not in PROVIDERS:
            raise ValueError(f"PROVIDER must be one of {PROVIDERS}, got {self.provider!r}")
        if not self.model:
            raise ValueError("MODEL is not set")
        if self.provider in ('openai', 'anthropic') and not self.api_key:
            raise ValueError(f"An API key is required for provider {self.provider!r}")
        return self

    def public(self):
        """Config safe to store in the run log (no secrets)."""
        d = asdict(self)
        d['api_key'] = '***' if self.api_key else ''
        d['base_url'] = self.base_url or DEFAULT_BASE_URLS[self.provider]
        return d


def make_llm(cfg: LLMConfig):
    cfg.validate()
    extra = {} if cfg.temperature is None else {'temperature': cfg.temperature}
    if cfg.provider == 'anthropic':
        from langchain_anthropic import ChatAnthropic
        # Pass base_url explicitly so a stray ANTHROPIC_BASE_URL in the environment doesn't silently reroute calls.
        return ChatAnthropic(model=cfg.model, api_key=cfg.api_key, max_tokens=cfg.max_tokens,
                             base_url=cfg.base_url or DEFAULT_BASE_URLS['anthropic'],
                             timeout=cfg.timeout, max_retries=2, **extra)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=cfg.model,
                      api_key=cfg.api_key or 'not-needed',  # local servers ignore it but the client requires one
                      base_url=cfg.base_url or DEFAULT_BASE_URLS[cfg.provider],
                      max_tokens=cfg.max_tokens, timeout=cfg.timeout, max_retries=2, **extra)


def response_text(msg):
    """Normalise message content: some providers return a list of content blocks instead of a string."""
    c = getattr(msg, 'content', msg)
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return ''.join(b if isinstance(b, str) else (b.get('text', '') if isinstance(b, dict) else '') for b in c)
    return str(c)


def ping(cfg: LLMConfig):
    """Cheap connectivity check used by the UI's 'Test connection' button."""
    from langchain_core.messages import HumanMessage
    return response_text(make_llm(cfg).invoke([HumanMessage(content='Reply with the single word: OK')]))


# ---------------------------------------------------------------- call logging, token accounting, context limits
_CONTEXT_PATTERNS = ('context length', 'context_length', 'context window', 'maximum context', 'prompt is too long', 'too many tokens',
                     'token limit', 'tokens exceed', 'request too large', 'reduce the length', 'input is too long', 'maximum prompt length',
                     'exceeds the model', 'string too long', 'payload too large')
STORE_LIMIT = 200_000     # characters of one call's input / output kept in the log


def is_context_error(exc) -> bool:
    """True when a provider error means 'the prompt does not fit' (wording differs per provider and server)."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(p in text for p in _CONTEXT_PATTERNS) or getattr(exc, 'status_code', None) == 413


def estimate_tokens(text) -> int:
    return max(1, len(text or '') // 4)


def message_dicts(msgs):
    return [{'role': {'system': 'system', 'human': 'user', 'ai': 'assistant'}.get(getattr(m, 'type', ''), getattr(m, 'type', 'user')),
             'content': response_text(m)} for m in msgs]


def usage_of(msg):
    """(prompt, completion, total) tokens reported by the provider for this reply, or None when it reported nothing."""
    u = getattr(msg, 'usage_metadata', None)
    if not isinstance(u, dict):
        meta = getattr(msg, 'response_metadata', None) or {}
        raw = meta.get('token_usage') or meta.get('usage') or {}
        u = {'input_tokens': raw.get('prompt_tokens', raw.get('input_tokens')), 'output_tokens': raw.get('completion_tokens', raw.get('output_tokens'))}
    p, c = u.get('input_tokens'), u.get('output_tokens')
    if p is None and c is None:
        return None
    p, c = int(p or 0), int(c or 0)
    return p, c, int(u.get('total_tokens') or p + c)


class TrackedLLM:
    """Wraps a chat model: every call is written to the workspace's llm_calls table (full input and output, tokens,
    latency, errors). `call()` returns the reply text and re-raises provider errors after logging them."""

    def __init__(self, llm, cfg, ws=None, agent='swarm'):
        self.llm, self.cfg, self.ws, self.agent = llm, cfg, ws, agent

    def call(self, msgs, purpose='call', step=None):
        t0 = time.time()
        dicts = message_dicts(msgs)
        inp = json.dumps(dicts, ensure_ascii=False)
        self._raw_in = sum(len(d['content']) for d in dicts)
        try:
            reply = self.llm.invoke(msgs)
        except Exception as e:
            self._log(step, purpose, inp, '', None, t0, 'error', f"{type(e).__name__}: {e}"[:1500])
            raise
        text = response_text(reply)
        self._log(step, purpose, inp, text, usage_of(reply), t0, 'ok', None)
        return text

    def _log(self, step, purpose, inp, out, usage, t0, status, error):
        if self.ws is None:
            return
        estimated = usage is None
        p, c, t = usage or (max(1, self._raw_in // 4), estimate_tokens(out), 0)
        try:
            self.ws.log_llm_call(self.agent, step, purpose, getattr(self.cfg, 'model', ''), inp[:STORE_LIMIT], out[:STORE_LIMIT],
                                 p, c, t or p + c, int((time.time() - t0) * 1000), status, error, estimated, len(inp), len(out))
        except Exception:
            pass          # accounting must never break a run
