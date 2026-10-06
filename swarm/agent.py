import json, re, time
from typing import Any, Optional, TypedDict
from pydantic import BaseModel, Field, ValidationError, field_validator
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from .llm import make_llm, response_text

SYSTEM = '''You are one of several identical autonomous agents working on a common objective. There is NO coordinator, planner, leader, assigned role, or direct agent-to-agent chat. A shared workspace (message board + artifacts) exists; other agents may have published there. Using it is optional. Decide independently what action is most useful next.

Your tools, used through fields of your JSON reply:
- data_query: run an exact computation on the dataset (if one is provided). Results arrive at your next step and are also saved as a public artifact. Never invent numbers; compute them.
- artifact_to_read: an artifact ID from the index. Its full content arrives at your next step.
- board_message: a short public note to the shared board.
- artifact_name + artifact_content: publish a substantial finding (with the numbers that support it).

Reply with ONE JSON object and nothing else. Do not reveal private chain-of-thought; audit_summary is a brief high-level account of what you observed, what you did and why.'''

QUERY_HELP = '''data_query format (all keys optional except what you need):
{"filters": [{"column": "Country", "op": "==", "value": "France"}],      ops: == != > >= < <= in "not in" contains
 "derive": [{"name": "unit_margin", "left": "Sale Price", "op": "-", "right": "Manufacturing Price"}],  ops: + - * / ; right may be a number
 "group_by": ["Segment", "Product"],
 "metrics": [{"column": "Profit", "agg": "sum"}, {"column": "Sales", "agg": "sum"}],  aggs: sum mean median min max count nunique std ; output names are <column>_<agg>
 "ratios": [{"name": "margin", "numerator": "Profit_sum", "denominator": "Sales_sum"}],
 "sort_by": "margin", "ascending": false, "limit": 20}
or {"correlation": ["Units Sold", "Discounts", "Profit"]}'''

REPLY_SCHEMA = '''{"status": "continue" | "done",
 "action": "<short label of what you are doing>",
 "audit_summary": "<1-3 sentences>",
 "confidence": <0..1>,
 "data_query": <object or null>,
 "artifact_to_read": <integer id or null>,
 "board_message": <string or null>,
 "artifact_name": <string or null>,
 "artifact_content": <string or null>}'''


class Decision(BaseModel):
    status: str = 'continue'
    action: str = ''
    audit_summary: str = ''
    confidence: Optional[float] = None
    data_query: Optional[dict] = None
    artifact_to_read: Optional[int] = None
    board_message: Optional[str] = None
    artifact_name: Optional[str] = None
    artifact_content: Optional[str] = None

    @field_validator('status', mode='before')
    @classmethod
    def _status(cls, v):
        return 'done' if str(v).strip().lower() in ('done', 'finished', 'complete', 'completed') else 'continue'

    @field_validator('action', 'audit_summary', 'board_message', 'artifact_name', 'artifact_content', mode='before')
    @classmethod
    def _to_text(cls, v):
        if v is None or isinstance(v, str):
            return v
        return json.dumps(v, ensure_ascii=False, default=str)

    @field_validator('artifact_to_read', mode='before')
    @classmethod
    def _to_int(cls, v):
        if v in (None, '', 'null'):
            return None
        m = re.search(r'\d+', str(v))
        return int(m.group(0)) if m else None

    @field_validator('confidence', mode='before')
    @classmethod
    def _conf(cls, v):
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator('data_query', mode='before')
    @classmethod
    def _query(cls, v):
        if isinstance(v, str):
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                return None
        return v if isinstance(v, dict) and v else None


def extract_json(text):
    """Pull the first JSON object out of a model reply (handles ```json fences and surrounding prose)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.S)  # local reasoning models
    fence = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.S)
    candidates = [fence.group(1)] if fence else []
    candidates += [text[i:] for i, ch in enumerate(text) if ch == '{']
    dec = json.JSONDecoder()
    for c in candidates:
        try:
            obj, _ = dec.raw_decode(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("no JSON object found in model reply")


def _clip(s, n):
    s = s or ''
    return s if len(s) <= n else s[:n] + f'... [truncated, {len(s) - n} more chars]'


class State(TypedDict, total=False):
    objective: str
    agent_id: str
    step: int
    max_steps: int
    done: bool
    history: list
    inbox: list
    errors: int
    decision: dict


class Agent:
    def __init__(self, agent_id, workspace, llm_cfg, dataset=None, max_errors=2):
        self.id = agent_id
        self.ws = workspace
        self.dataset = dataset
        self.max_errors = max_errors
        self.llm = make_llm(llm_cfg)
        self._graph = self._build()

    def _build(self):
        g = StateGraph(State)
        g.add_node('act', self.act)
        g.set_entry_point('act')
        g.add_conditional_edges('act', lambda s: END if s['done'] or s['step'] >= s['max_steps'] else 'act')
        return g.compile()

    def _prompt(self, s):
        final = s['step'] + 1 >= s['max_steps']
        parts = [f"OBJECTIVE:\n{s['objective']}",
                 f"AGENT ID: {self.id}   STEP: {s['step'] + 1} of {s['max_steps']}"]
        if self.dataset is not None:
            parts.append(f"DATASET '{self.dataset.name}' (already cleaned as noted; query it with data_query):\n{self.dataset.profile}\n\n{QUERY_HELP}")
        board = self.ws.recent_board(30)
        index = self.ws.artifact_index(40)
        parts.append("SHARED BOARD (latest):\n" + ("\n".join(f"#{i} [{a}] {c}" for i, a, k, c in board) or "(empty)"))
        parts.append("SHARED ARTIFACT INDEX (id, author, kind, name, preview; read one to get full content):\n" +
                     ("\n".join(f"{i} | {a} | {k} | {n} | {(p or '').replace(chr(10), ' ')}" for i, a, n, k, p in index) or "(empty)"))
        if s.get('history'):
            parts.append("YOUR PREVIOUS STEPS:\n" + "\n".join(f"- step {h['step']}: {h['action']} — {h['summary']}" for h in s['history'][-8:]))
        if s.get('inbox'):
            parts.append("RESULTS FROM YOUR LAST STEP:\n" + "\n\n".join(s['inbox']))
        if final:
            parts.append("THIS IS YOUR FINAL STEP: publish your best final answer as an artifact (artifact_name + artifact_content, "
                         "citing the computed numbers) and set status to done.")
        parts.append("Reply with exactly one JSON object:\n" + REPLY_SCHEMA)
        return "\n\n".join(parts)

    def _decide(self, prompt):
        msgs = [SystemMessage(content=SYSTEM), HumanMessage(content=prompt)]
        txt = response_text(self.llm.invoke(msgs))
        try:
            return Decision.model_validate(extract_json(txt)), txt
        except (ValueError, ValidationError) as e:
            # One repair attempt: show the model its own reply and the error.
            msgs += [HumanMessage(content=f"Your reply could not be parsed ({e}). Your reply was:\n{_clip(txt, 1500)}\n\n"
                                          f"Reply again with ONLY a valid JSON object matching:\n{REPLY_SCHEMA}")]
            txt = response_text(self.llm.invoke(msgs))
            return Decision.model_validate(extract_json(txt)), txt

    def act(self, s):
        step = s['step'] + 1
        try:
            d, _ = self._decide(self._prompt(s))
        except Exception as e:
            errors = s.get('errors', 0) + 1
            self.ws.trace(self.id, 'error', {'step': step, 'error': f"{type(e).__name__}: {_clip(str(e), 500)}"})
            if errors <= self.max_errors:
                time.sleep(2 * errors)
            return {**s, 'step': step, 'errors': errors, 'done': errors > self.max_errors}

        self.ws.trace(self.id, 'decision', {'step': step, **d.model_dump(include={'status', 'action', 'audit_summary', 'confidence', 'artifact_to_read', 'data_query'})})
        inbox = []
        if d.artifact_to_read is not None:
            art = self.ws.access(self.id, d.artifact_to_read)
            if art is None:
                inbox.append(f"Artifact {d.artifact_to_read} does not exist.")
            else:
                inbox.append(f"ARTIFACT {art[0]} by {art[2]} — {art[3]}:\n{_clip(art[5], 6000)}")
        if d.data_query:
            if self.dataset is None:
                inbox.append("data_query ignored: no dataset was provided for this run.")
            else:
                spec = json.dumps(d.data_query, default=str)
                try:
                    result = self.dataset.query(d.data_query)
                    aid = self.ws.artifact(self.id, f"query: {_clip(d.action or spec, 80)}", f"QUERY {spec}\n\n{result}", kind='query_result')
                    self.ws.trace(self.id, 'query', {'step': step, 'artifact_id': aid})
                    inbox.append(f"QUERY {spec}\n{result}")
                except Exception as e:
                    self.ws.trace(self.id, 'query_error', {'step': step, 'query': d.data_query, 'error': str(e)})
                    inbox.append(f"QUERY {spec} FAILED: {e}")
        if d.board_message:
            self.ws.board(self.id, 'note', d.board_message)
        if d.artifact_content:
            kind = 'final_answer' if d.status == 'done' or step >= s['max_steps'] else 'finding'
            aid = self.ws.artifact(self.id, d.artifact_name or f'{self.id}_step_{step}', d.artifact_content, kind=kind)
            self.ws.trace(self.id, 'artifact_created', {'step': step, 'artifact_id': aid, 'kind': kind})

        history = (s.get('history') or []) + [{'step': step, 'action': d.action, 'summary': _clip(d.audit_summary, 300)}]
        return {**s, 'decision': d.model_dump(), 'done': d.status == 'done', 'step': step,
                'history': history, 'inbox': inbox, 'errors': 0}

    def run(self, objective, max_steps):
        state = {'objective': objective, 'agent_id': self.id, 'step': 0, 'max_steps': max_steps,
                 'done': False, 'history': [], 'inbox': [], 'errors': 0}
        return self._graph.invoke(state, {'recursion_limit': max_steps + 5})


def run_agent(agent_id, objective, workspace, llm_cfg, steps=6, dataset=None, start_delay=0.0):
    if start_delay:
        time.sleep(start_delay)
    workspace.trace(agent_id, 'run_started', {'max_steps': steps})
    try:
        state = Agent(agent_id, workspace, llm_cfg, dataset).run(objective, steps)
    except Exception as e:
        workspace.trace(agent_id, 'run_failed', {'error': f"{type(e).__name__}: {e}"})
        return {'agent_id': agent_id, 'step': 0, 'done': False, 'error': str(e)}
    workspace.trace(agent_id, 'run_finished', {'steps': state['step'], 'done': state['done'], 'errors': state.get('errors', 0)})
    return state
