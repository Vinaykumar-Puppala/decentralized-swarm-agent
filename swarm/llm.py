"""Provider-agnostic chat model factory.

PROVIDER   openai | anthropic | litellm | local
MODEL      model name as the provider expects it (e.g. gpt-4o-mini, claude-sonnet-4-5, ollama model tag)
API_KEY    key for the provider (falls back to OPENAI_API_KEY / ANTHROPIC_API_KEY / LITELLM_API_KEY)
BASE_URL   optional endpoint override (falls back to OPENAI_BASE_URL)

`local` and `litellm` talk to any OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp, LiteLLM proxy).
"""
import os
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
