import json, re, time
from typing import Any, Optional, TypedDict
from pydantic import BaseModel, Field, ValidationError, field_validator
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from .llm import TrackedLLM, is_context_error, make_llm
from .workspace import missing_ranges

SYSTEM = '''You are one of several identical autonomous agents working on a common objective. There is NO coordinator, planner, leader, assigned role, or direct agent-to-agent chat. A shared workspace (message board + artifacts) exists; other agents may have published there. Using it is optional. Decide independently what action is most useful next.

Your tools, used through fields of your JSON reply:
- data_query: run an exact aggregate computation on the data (if provided). Results arrive at your next step and are also saved as a public artifact. Never invent numbers; compute them.
- data_rows: read the actual rows of a table, a page at a time. Aggregates hide row-level problems and patterns, so when data is provided do not rely on data_query alone: read rows to find outliers, duplicates, inconsistent or suspicious values, rule violations and individual cases that explain the aggregates. The workspace shows which row ids the swarm has already reviewed; prefer unreviewed rows so that, together, the agents cover every row. Cite row ids (_row) in your findings.
- artifact_to_read: an artifact ID from the index. Its full content arrives at your next step.
- board_message: a short public note to the shared board.
- artifact_name + artifact_content: publish a substantial finding (with the numbers that support it).
- scratchpad_note: append a note to your own scratchpad: private working notes to yourself that persist across your steps (hypotheses, what you ruled out, what to do next). Optional.
{extra_tools}
Reply with ONE JSON object and nothing else. Do not reveal private chain-of-thought. Instead, every step you write two things for the record: audit_summary, a brief account of what you observed and did, and rationale, a deliberate written explanation of why you chose this step: what you observed, which alternatives you considered, why this one, and what you expect it to show. Be specific (numbers, row ids, column names). The rationale is a considered explanation for readers, not a transcript of your private thinking.'''

LOG_TOOL = '''- read_log: look at the shared ACTIVITY LOG, which records for every agent step what was tried, why, and what came back (including failures). Value: {"agent": "agent-2", "limit": 10} (both keys optional) or true. Entirely optional: use it only if you decide it will help you. Many steps will not need it, and you are never expected to read it.'''
SCRATCH_TOOL = '''- read_scratchpad: the id of another agent (e.g. "agent-3"), to read that agent's scratchpad notes. Entirely optional, like the log.'''
VISIBILITY = ('isolated', 'log', 'full')   # isolated: no peer reading | log: activity log readable | full: log + scratchpads readable


def system_prompt(visibility='full'):
    extra = []
    if visibility in ('log', 'full'):
        extra.append(LOG_TOOL)
    if visibility == 'full':
        extra.append(SCRATCH_TOOL)
    return SYSTEM.replace('{extra_tools}', "\n".join(extra) + ("\n" if extra else ""))


def reply_schema(visibility='full'):
    opt = ['"scratchpad_note": <string or null>']
    if visibility in ('log', 'full'):
        opt.append('"read_log": <object, true or null>')
    if visibility == 'full':
        opt.append('"read_scratchpad": <agent id or null>')
    return REPLY_SCHEMA.replace('{optional}', ',\n '.join(opt))

QUERY_HELP = '''data_query format (all keys optional except what you need; with several tables set "table"; tables are separate and cannot be joined):
{"table": "sales",
 "filters": [{"column": "Country", "op": "==", "value": "France"}],      ops: == != > >= < <= in "not in" contains
 "derive": [{"name": "unit_margin", "left": "Sale Price", "op": "-", "right": "Manufacturing Price"}],  ops: + - * / ; right may be a number
 "group_by": ["Segment", "Product"],
 "metrics": [{"column": "Profit", "agg": "sum"}, {"column": "Sales", "agg": "sum"}],  aggs: sum mean median min max count nunique std ; output names are <column>_<agg>
 "ratios": [{"name": "margin", "numerator": "Profit_sum", "denominator": "Sales_sum"}],
 "sort_by": "margin", "ascending": false, "limit": 20}
or {"correlation": ["Units Sold", "Discounts", "Profit"]}

data_rows format (reads raw rows; at most 100 per call; keep "columns" short to save space):
{"table": "sales", "offset": 0, "limit": 50, "columns": ["Segment", "Country", "Profit"],
 "filters": [...same as data_query...], "derive": [...same...], "sort_by": "Profit", "ascending": true}
Offsets page through the matching rows; the first CSV column _row is the stable row id.'''

REPLY_SCHEMA = '''{"status": "continue" | "done",
 "action": "<short label of what you are doing>",
 "audit_summary": "<1-3 sentences>",
 "rationale": "<3-6 sentences: what you observed, alternatives considered, why this step, what you expect it to show>",
 "confidence": <0..1>,
 "data_query": <object or null>,
 "data_rows": <object or null>,
 "artifact_to_read": <integer id or null>,
 "board_message": <string or null>,
 "artifact_name": <string or null>,
 "artifact_content": <string or null>,
 {optional}}'''


class Decision(BaseModel):
    status: str = 'continue'
    action: str = ''
    audit_summary: str = ''
    confidence: Optional[float] = None
    data_query: Optional[dict] = None
    data_rows: Optional[dict] = None
    artifact_to_read: Optional[int] = None
    board_message: Optional[str] = None
    artifact_name: Optional[str] = None
    artifact_content: Optional[str] = None
    rationale: Optional[str] = ''
    scratchpad_note: Optional[str] = None
    read_log: Optional[dict] = None            # None = not requested; {} = "the whole log"
    read_scratchpad: Optional[str] = None      # another agent's id

    @field_validator('read_log', mode='before')
    @classmethod
    def _read_log(cls, v):
        if v is None or v is False or v in ('', 'null', 'false', 'False'):
            return None
        if isinstance(v, dict):
            out = {}
            if v.get('agent'):
                out['agent'] = str(v['agent']).strip()
            try:
                if v.get('limit') is not None:
                    out['limit'] = max(1, min(int(v['limit']), 30))
            except (TypeError, ValueError):
                pass
            return out
        if isinstance(v, str) and v.strip().lower() not in ('true', 'yes', '1'):
            return {'agent': v.strip()}
        return {}

    @field_validator('read_scratchpad', mode='before')
    @classmethod
    def _read_scratch(cls, v):
        if v in (None, '', 'null', False, 'false'):
            return None
        if isinstance(v, int):
            return f'agent-{v}'
        v = str(v).strip()
        return f'agent-{v}' if v.isdigit() else v

    @field_validator('status', mode='before')
    @classmethod
    def _status(cls, v):
        return 'done' if str(v).strip().lower() in ('done', 'finished', 'complete', 'completed') else 'continue'

    @field_validator('action', 'audit_summary', 'board_message', 'artifact_name', 'artifact_content', 'rationale', 'scratchpad_note', mode='before')
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

    @field_validator('data_query', 'data_rows', mode='before')
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


def format_log_entries(rows):
    """Activity-log rows -> readable text for an agent that chose to read the log."""
    out = []
    for _id, _ts, agent, step, kind, action, summary, rationale, conf, tools, _target in rows:
        if kind == 'error':
            out.append(f"#{_id} {agent} step {step}: {action}: {_clip(summary, 300)}")
            continue
        head = f"#{_id} {agent} step {step}: {action or '(no label)'}" + (f" (confidence {conf:.2f})" if conf is not None else '')
        lines = [head]
        if rationale or summary:
            lines.append(f"  why: {_clip(rationale or summary, 900)}")
        for t in json.loads(tools or '[]'):
            args = json.dumps(t.get('args'), default=str)
            lines.append(f"  tried: {t['tool']} {_clip(args, 200)} -> {'ok' if t.get('ok', True) else 'FAILED'}: {t.get('outcome', '')}")
        out.append("\n".join(lines))
    return _clip("\n".join(out), 8000)


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
    def __init__(self, agent_id, workspace, llm_cfg, dataset=None, max_errors=2, visibility='full'):
        if visibility not in VISIBILITY:
            raise ValueError(f"visibility must be one of {VISIBILITY}, got {visibility!r}")
        self.id = agent_id
        self.ws = workspace
        self.dataset = dataset
        self.max_errors = max_errors
        self.visibility = visibility
        self.system = system_prompt(visibility)
        self.schema = reply_schema(visibility)
        self.llm = TrackedLLM(make_llm(llm_cfg), llm_cfg, workspace, agent_id)
        self._graph = self._build()

    def _peer_pointers(self):
        """Pointers only, never content: the agent has to ask (read_log / read_scratchpad) to see anything of its peers."""
        out = []
        mine = self.ws.scratch_notes(self.id)
        if mine:
            out.append("YOUR SCRATCHPAD (your own notes, only you wrote these):\n" + _clip("\n".join(f"- (step {st}) {n}" for _, _, st, n in mine), 3000))
        if self.visibility in ('log', 'full'):
            counts = self.ws.activity_overview()
            out.append("SHARED ACTIVITY LOG: " + (
                f"{sum(counts.values())} step entries so far ({', '.join(f'{a}: {n}' for a, n in sorted(counts.items()))}). "
                if counts else "no entries yet. ") +
                "It records what each agent tried, why, and what came back. You are free to ignore it; use read_log only if you wish.")
        if self.visibility == 'full':
            others = {a: v for a, v in self.ws.scratch_overview().items() if a != self.id}
            if others:
                out.append("OTHER AGENTS' SCRATCHPADS (notes each agent keeps for itself): " +
                           ", ".join(f"{a} ({n} notes)" for a, (n, _) in sorted(others.items())) +
                           ". You may read one with read_scratchpad if you wish; you never have to.")
        return out

    def _build(self):
        g = StateGraph(State)
        g.add_node('act', self.act)
        g.set_entry_point('act')
        g.add_conditional_edges('act', lambda s: END if s['done'] or s['step'] >= s['max_steps'] else 'act')
        return g.compile()

    def _coverage_text(self, max_gaps=6):
        """Public, shared view of which rows the swarm has already read (so agents can pick unread ones)."""
        lines = []
        for name, ds in self.dataset.tables.items():
            total = len(ds.df)
            seen, mine = self.ws.coverage(name), self.ws.coverage(name, self.id)
            n_seen, n_mine = sum(b - a + 1 for a, b in seen), sum(b - a + 1 for a, b in mine)
            gaps = missing_ranges(seen, total)
            shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in gaps[:max_gaps])
            tail = f" (+{len(gaps) - max_gaps} more ranges)" if len(gaps) > max_gaps else ""
            lines.append(f"- '{name}': {n_seen}/{total} rows ({100 * n_seen / total:.0f}%) read by the swarm, {n_mine} by you. "
                         f"Unread row ids: {shown + tail if gaps else 'none, every row has been read'}")
        return "ROW COVERAGE (rows read so far with data_rows by any agent):\n" + "\n".join(lines)

    def _prompt(self, s, compact=False):
        """compact=True is the retry after a context-limit error: shorter board / index / history and clipped tool results."""
        final = s['step'] + 1 >= s['max_steps']
        parts = [f"OBJECTIVE:\n{s['objective']}",
                 f"AGENT ID: {self.id}   STEP: {s['step'] + 1} of {s['max_steps']}"]
        if self.dataset is not None:
            parts.append(f"DATA: {self.dataset.name} (already cleaned as noted; analyse it with data_query and data_rows):\n{self.dataset.profile}\n\n{QUERY_HELP}")
            parts.append(self._coverage_text())
        board = self.ws.recent_board(8 if compact else 30)
        index = self.ws.artifact_index(10 if compact else 40, 80 if compact else 160)
        parts.append("SHARED BOARD (latest):\n" + ("\n".join(f"#{i} [{a}] {c}" for i, a, k, c in board) or "(empty)"))
        parts.append("SHARED ARTIFACT INDEX (id, author, kind, name, preview; read one to get full content):\n" +
                     ("\n".join(f"{i} | {a} | {k} | {n} | {(p or '').replace(chr(10), ' ')}" for i, a, n, k, p in index) or "(empty)"))
        parts += self._peer_pointers()
        if s.get('history'):
            parts.append("YOUR PREVIOUS STEPS:\n" + "\n".join(f"- step {h['step']}: {h['action']} — {h['summary']}" for h in s['history'][-(3 if compact else 8):]))
        if s.get('inbox'):
            inbox = "\n\n".join(s['inbox'])
            parts.append("RESULTS FROM YOUR LAST STEP:\n" + (_clip(inbox, 6000) + "\n[results shortened to fit the model's context window]" if compact and len(inbox) > 6000 else inbox))
        if final:
            parts.append("THIS IS YOUR FINAL STEP: publish your best final answer as an artifact (artifact_name + artifact_content, "
                         "citing the computed numbers) and set status to done.")
        parts.append("Reply with exactly one JSON object:\n" + self.schema)
        return "\n\n".join(parts)

    def _decide(self, prompt, step=None):
        msgs = [SystemMessage(content=self.system), HumanMessage(content=prompt)]
        txt = self.llm.call(msgs, 'decision', step)
        try:
            return Decision.model_validate(extract_json(txt)), txt
        except (ValueError, ValidationError) as e:
            # One repair attempt: show the model its own reply and the error.
            msgs += [HumanMessage(content=f"Your reply could not be parsed ({e}). Your reply was:\n{_clip(txt, 1500)}\n\n"
                                          f"Reply again with ONLY a valid JSON object matching:\n{self.schema}")]
            txt = self.llm.call(msgs, 'repair', step)
            return Decision.model_validate(extract_json(txt)), txt

    def act(self, s):
        step = s['step'] + 1
        if self.ws.is_cancelled():
            self.ws.trace(self.id, 'cancelled', {'step': step})
            return {**s, 'step': s['step'], 'done': True}
        try:
            self.ws.trace(self.id, 'step_started', {'step': step})
            try:
                d, _ = self._decide(self._prompt(s), step)
            except Exception as e:
                if not is_context_error(e):
                    raise
                # The prompt did not fit the model's context window: say so, then retry once with a shortened prompt.
                self.ws.trace(self.id, 'context_retry', {'step': step, 'error': _clip(str(e), 300)})
                d, _ = self._decide(self._prompt(s, compact=True), step)
        except Exception as e:
            errors = s.get('errors', 0) + 1
            msg = f"{type(e).__name__}: {_clip(str(e), 500)}"
            self.ws.trace(self.id, 'error', {'step': step, 'error': msg})
            self.ws.log_activity(self.id, step, 'error', action='model call failed', summary=msg)
            if errors <= self.max_errors:
                time.sleep(2 * errors)
            return {**s, 'step': step, 'errors': errors, 'done': errors > self.max_errors}

        self.ws.trace(self.id, 'decision', {'step': step, **d.model_dump(include={
            'status', 'action', 'audit_summary', 'rationale', 'confidence', 'artifact_to_read', 'data_query', 'data_rows',
            'read_log', 'read_scratchpad'}), 'scratchpad_note': _clip(d.scratchpad_note, 300) if d.scratchpad_note else None})
        inbox, tools = [], []          # tools: what this step did, for the shared activity log

        def record(tool, args, outcome, ok=True):
            tools.append({'tool': tool, 'args': args, 'outcome': _clip(outcome, 200), 'ok': ok})

        if d.artifact_to_read is not None:
            art = self.ws.access(self.id, d.artifact_to_read)
            if art is None:
                inbox.append(f"Artifact {d.artifact_to_read} does not exist.")
                record('read_artifact', {'artifact_id': d.artifact_to_read}, 'not found', False)
            else:
                inbox.append(f"ARTIFACT {art[0]} by {art[2]} — {art[3]}:\n{_clip(art[5], 6000)}")
                record('read_artifact', {'artifact_id': art[0]}, f"artifact #{art[0]} by {art[2]}")
        if d.data_query:
            if self.dataset is None:
                inbox.append("data_query ignored: no dataset was provided for this run.")
                record('data_query', d.data_query, 'no dataset in this run', False)
            else:
                spec = json.dumps(d.data_query, default=str)
                try:
                    result = self.dataset.query(d.data_query)
                    aid = self.ws.artifact(self.id, f"query: {_clip(d.action or spec, 80)}", f"QUERY {spec}\n\n{result}", kind='query_result')
                    self.ws.trace(self.id, 'query', {'step': step, 'artifact_id': aid})
                    inbox.append(f"QUERY {spec}\n{result}")
                    record('data_query', d.data_query, f"{result.splitlines()[0]} (saved as artifact #{aid})")
                except Exception as e:
                    self.ws.trace(self.id, 'query_error', {'step': step, 'tool': 'data_query', 'query': d.data_query, 'error': str(e)})
                    inbox.append(f"QUERY {spec} FAILED: {e}")
                    record('data_query', d.data_query, f"FAILED: {e}", False)
        if d.data_rows:
            if self.dataset is None:
                inbox.append("data_rows ignored: no dataset was provided for this run.")
                record('data_rows', d.data_rows, 'no dataset in this run', False)
            else:
                spec = json.dumps(d.data_rows, default=str)
                try:
                    text, table, ids = self.dataset.read_rows(d.data_rows)
                    self.ws.log_rows(self.id, table, ids)
                    self.ws.trace(self.id, 'rows_viewed', {'step': step, 'table': table, 'count': len(ids),
                                                           'first': ids[0] if ids else None, 'last': ids[-1] if ids else None})
                    inbox.append(f"DATA_ROWS {spec} (table '{table}')\n{text}")
                    record('data_rows', d.data_rows, f"{len(ids)} rows of '{table}'" + (f" (row ids {ids[0]}-{ids[-1]})" if ids else ''))
                except Exception as e:
                    self.ws.trace(self.id, 'query_error', {'step': step, 'tool': 'data_rows', 'query': d.data_rows, 'error': str(e)})
                    inbox.append(f"DATA_ROWS {spec} FAILED: {e}")
                    record('data_rows', d.data_rows, f"FAILED: {e}", False)

        # Peer reading is always the agent's own choice: nothing here runs unless the reply asked for it.
        if d.read_log is not None:
            self._read_log(d.read_log, step, inbox, record)
        if d.read_scratchpad:
            self._read_scratchpad(d.read_scratchpad, step, inbox, record)
        if d.scratchpad_note:
            self.ws.scratch_add(self.id, step, _clip(d.scratchpad_note, 4000))
            record('scratchpad_note', {'chars': len(d.scratchpad_note)}, 'saved to your scratchpad')

        if d.board_message:
            self.ws.board(self.id, 'note', d.board_message)
            record('board_message', {'text': _clip(d.board_message, 120)}, 'posted to the board')
        if d.artifact_content:
            kind = 'final_answer' if d.status == 'done' or step >= s['max_steps'] else 'finding'
            aid = self.ws.artifact(self.id, d.artifact_name or f'{self.id}_step_{step}', d.artifact_content, kind=kind)
            self.ws.trace(self.id, 'artifact_created', {'step': step, 'artifact_id': aid, 'kind': kind})
            record('publish_artifact', {'name': d.artifact_name, 'kind': kind}, f"published as artifact #{aid}")

        self.ws.log_activity(self.id, step, 'step', action=d.action, summary=d.audit_summary, rationale=d.rationale or '',
                             confidence=d.confidence, tools=tools)
        history = (s.get('history') or []) + [{'step': step, 'action': d.action, 'summary': _clip(d.audit_summary, 300)}]
        return {**s, 'decision': d.model_dump(), 'done': d.status == 'done', 'step': step,
                'history': history, 'inbox': inbox, 'errors': 0}

    # ---- optional peer reading: each is a tool the agent asked for, and each is itself recorded in the shared log
    def _read_log(self, spec, step, inbox, record):
        if self.visibility == 'isolated':
            inbox.append("read_log is not available in this run.")
            record('read_log', spec, 'not available in this run', False)
            return
        who, limit = spec.get('agent'), spec.get('limit', 10)
        rows = self.ws.activity_entries(agent=who, limit=limit, kinds=('step', 'error'))
        scope = f" of {who}" if who else ''
        inbox.append(f"ACTIVITY LOG{scope} ({len(rows)} entries, oldest first):\n" + (format_log_entries(rows) or "(no entries yet)"))
        self.ws.log_activity(self.id, step, 'read', action=f"read the activity log{scope}", summary=f"{len(rows)} entries", target=f"log:{who or 'all'}")
        self.ws.trace(self.id, 'activity_read', {'step': step, 'target': who or 'all', 'count': len(rows)})
        record('read_log', spec, f"{len(rows)} entries")

    def _read_scratchpad(self, target, step, inbox, record):
        if target != self.id and self.visibility != 'full':
            inbox.append("Other agents' scratchpads are private in this run.")
            record('read_scratchpad', {'agent': target}, 'private in this run', False)
            return
        notes = self.ws.scratch_notes(target)
        body = "\n".join(f"- (step {st}) {n}" for _, _, st, n in notes)
        inbox.append(f"SCRATCHPAD of {target} ({len(notes)} notes):\n" + (_clip(body, 6000) or "(empty)"))
        if target != self.id:     # reading your own notes is not peer reading
            self.ws.log_activity(self.id, step, 'read', action=f"read {target}'s scratchpad", summary=f"{len(notes)} notes", target=f"scratchpad:{target}")
            self.ws.trace(self.id, 'scratchpad_read', {'step': step, 'target': target, 'count': len(notes)})
        record('read_scratchpad', {'agent': target}, f"{len(notes)} notes")

    def run(self, objective, max_steps):
        state = {'objective': objective, 'agent_id': self.id, 'step': 0, 'max_steps': max_steps,
                 'done': False, 'history': [], 'inbox': [], 'errors': 0}
        return self._graph.invoke(state, {'recursion_limit': max_steps + 5})


def run_agent(agent_id, objective, workspace, llm_cfg, steps=6, dataset=None, start_delay=0.0, visibility='full'):
    if start_delay:
        time.sleep(start_delay)
    workspace.trace(agent_id, 'run_started', {'max_steps': steps})
    try:
        state = Agent(agent_id, workspace, llm_cfg, dataset, visibility=visibility).run(objective, steps)
    except Exception as e:
        workspace.trace(agent_id, 'run_failed', {'error': f"{type(e).__name__}: {e}"})
        return {'agent_id': agent_id, 'step': 0, 'done': False, 'error': str(e)}
    workspace.trace(agent_id, 'run_finished', {'steps': state['step'], 'done': state['done'], 'errors': state.get('errors', 0)})
    return state
