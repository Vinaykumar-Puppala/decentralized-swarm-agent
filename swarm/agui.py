"""Translate a swarm run (rows in the workspace) into an AG-UI event stream (https://docs.ag-ui.com).

Mapping
  one experiment                      -> one AG-UI run (RUN_STARTED ... RUN_FINISHED / RUN_ERROR)
  each autonomous agent               -> a sub-agent run (SUBAGENT_STARTED/FINISHED/ERROR; every event it causes
                                         carries its subagentRunId, so the UI can attribute and group concurrent agents)
  an agent step                       -> STEP_STARTED / STEP_FINISHED (step-1, step-2, ...)
  audit summary of a decision         -> TEXT_MESSAGE_* (role assistant, name = agent)   [never hidden chain-of-thought]
  data_query / data_rows / read       -> TOOL_CALL_START / ARGS / END, then TOOL_CALL_RESULT
  shared board post / artifact        -> ACTIVITY_SNAPSHOT (activityType board_post / artifact)
  every model call (tokens, latency)  -> ACTIVITY_SNAPSHOT (llm_call); full input/output via GET /api/runs/{id}/llm-calls/{n}
  artifact access, agent errors       -> CUSTOM (swarm.access / swarm.agent_error / swarm.agent_cancelled)
  counters, per-agent status, row coverage -> STATE_SNAPSHOT, then STATE_DELTA (JSON Patch)
  post-hoc report                     -> TEXT_MESSAGE_* streamed in chunks (name = reporter)
"""
import json
from typing import Optional

import jsonpatch
from ag_ui.core import (ActivitySnapshotEvent, CustomEvent, RunErrorEvent, RunFinishedEvent, RunStartedEvent,
                        StateDeltaEvent, StateSnapshotEvent, StepFinishedEvent, StepStartedEvent,
                        SubagentErrorEvent, SubagentFinishedEvent, SubagentStartedEvent, TextMessageContentEvent,
                        TextMessageEndEvent, TextMessageStartEvent, ToolCallArgsEvent, ToolCallEndEvent,
                        ToolCallResultEvent, ToolCallStartEvent)

from .summary import ACTIVE, RunView

REPORT_CHUNK = 160
RESULT_LIMIT = 12000


def _clip(s, n=RESULT_LIMIT):
    s = s or ''
    return s if len(s) <= n else s[:n] + f'\n… [{len(s) - n} more characters]'


class AguiTranslator:
    def __init__(self, ws, thread_id, run_id):
        self.ws, self.thread_id, self.run_id = ws, thread_id, run_id
        self.view = RunView()
        self.state = None
        self.started, self.finished_agents, self.failed_agents = set(), set(), set()
        self.open_step = {}          # agent -> step name currently open
        self.pending_read = {}       # agent -> tool_call_id of an unanswered read_artifact call
        self.crashed = None
        self.done = False

    # ---------------- helpers
    def sub(self, agent):
        return f"{self.ws.run_id}:{agent}"

    @staticmethod
    def _ms(ts):
        return int(ts * 1000) if ts else None

    def _ensure_started(self, agent, ts, out):
        if agent in self.started:
            return
        self.started.add(agent)
        desc = 'post-hoc observer that summarises the finished run' if agent == 'reporter' else 'autonomous swarm agent'
        out.append(SubagentStartedEvent(subagent_run_id=self.sub(agent), name=agent, description=desc, timestamp=self._ms(ts)))

    def _close_step(self, agent, ts, out):
        step = self.open_step.pop(agent, None)
        if step:
            out.append(StepFinishedEvent(step_name=step, subagent_run_id=self.sub(agent), timestamp=self._ms(ts)))

    def _text(self, message_id, agent, text, ts, chunks=None):
        sid = self.sub(agent)
        yield TextMessageStartEvent(message_id=message_id, role='assistant', name=agent, subagent_run_id=sid, timestamp=self._ms(ts))
        for piece in chunks or [text]:
            yield TextMessageContentEvent(message_id=message_id, delta=piece, subagent_run_id=sid, timestamp=self._ms(ts))
        yield TextMessageEndEvent(message_id=message_id, subagent_run_id=sid, timestamp=self._ms(ts))

    def _tool_call(self, call_id, name, args, parent, agent, ts):
        sid = self.sub(agent)
        yield ToolCallStartEvent(tool_call_id=call_id, tool_call_name=name, parent_message_id=parent, subagent_run_id=sid, timestamp=self._ms(ts))
        yield ToolCallArgsEvent(tool_call_id=call_id, delta=json.dumps(args, default=str), subagent_run_id=sid, timestamp=self._ms(ts))
        yield ToolCallEndEvent(tool_call_id=call_id, subagent_run_id=sid, timestamp=self._ms(ts))

    def _tool_result(self, call_id, content, agent, ts):
        return ToolCallResultEvent(message_id=f"{call_id}-result", tool_call_id=call_id, content=_clip(content), role='tool',
                                   subagent_run_id=self.sub(agent), timestamp=self._ms(ts))

    # ---------------- lifecycle
    def begin(self):
        info = self.ws.run_info()
        self.state = self._build_state(info)
        return [RunStartedEvent(thread_id=self.thread_id, run_id=self.run_id),
                StateSnapshotEvent(snapshot=self.state)]

    def poll(self):
        """Events produced since the previous call. Reads the run status BEFORE the new rows: finish_run() is the last
        write of a run, so a final status means this read already contains every event of the run."""
        info = self.ws.run_info()
        out = []
        for e in self.view.update(self.ws):
            out += self._map(e)
        new_state = self._build_state(info)
        patch = jsonpatch.make_patch(self.state, new_state).patch
        if patch:
            out.append(StateDeltaEvent(delta=patch))
            self.state = new_state
        if info[4] not in ACTIVE:
            out += self._finalize(info)
        return out

    def _finalize(self, info):
        out, status, report = [], info[4], info[5]
        last = self.view.last_ts
        if report and 'reporter' in self.started:
            msg = 'report'
            chunks = [report[i:i + REPORT_CHUNK] for i in range(0, len(report), REPORT_CHUNK)]
            out += list(self._text(f"{self.run_id}-{msg}", 'reporter', report, last, chunks))
        for agent in sorted(self.started):
            self._close_step(agent, last, out)
            if agent not in self.finished_agents and agent not in self.failed_agents:
                if agent == 'reporter' and not report and status != 'cancelled':
                    out.append(SubagentErrorEvent(subagent_run_id=self.sub(agent), message='no report was produced', timestamp=self._ms(last)))
                elif agent == 'reporter' or status in ('finished', 'cancelled') or status.startswith('finished'):
                    out.append(SubagentFinishedEvent(subagent_run_id=self.sub(agent), result={'status': status}, outcome={'type': 'success'}, timestamp=self._ms(last)))
                else:
                    out.append(SubagentErrorEvent(subagent_run_id=self.sub(agent), message=f'run {status}', timestamp=self._ms(last)))
                self.finished_agents.add(agent)
        if status in ('crashed', 'interrupted') or self.crashed:
            out.append(RunErrorEvent(message=self.crashed or f'run {status}', code=status))
        elif status == 'cancelled':
            out.append(RunFinishedEvent(thread_id=self.thread_id, run_id=self.run_id, result={'status': status}, outcome={'type': 'cancelled'}))
        else:
            out.append(RunFinishedEvent(thread_id=self.thread_id, run_id=self.run_id,
                                        result={'status': status, 'report': report}, outcome={'type': 'success'}))
        self.done = True
        return out

    # ---------------- shared state
    def _build_state(self, info):
        st = self.view.status(self.ws, info)
        cfg = st['config']
        return {
            'swarmRunId': st['run_id'], 'status': st['status'], 'objective': st['objective'], 'startedAt': st['started'],
            'config': {'llm': {k: v for k, v in cfg.get('llm', {}).items() if k != 'api_key'}, 'nAgents': cfg.get('n_agents'), 'steps': cfg.get('steps'),
                       'dataset': cfg.get('dataset'), 'tables': cfg.get('tables', {})},
            'counts': st['counts'],
            'agents': {a['agent']: {'state': a['state'], 'steps': a['steps'], 'errors': a['errors'], 'rowsRead': a['rows_read'],
                                    'lastTs': self.view.agents[a['agent']]['last_ts'], 'action': a['action'], 'summary': a['summary']}
                       for a in st['agents']},
            'tokens': st['tokens'],
            'coverage': {t: {'total': c['total'], 'read': c['read'], 'ranges': c['ranges'], 'agents': c['agents']}
                         for t, c in st['coverage'].items()},
        }

    # ---------------- row -> events
    def _map(self, e):
        out, ts = [], e['ts']
        kind = e['type']
        if kind == 'board':
            out.append(ActivitySnapshotEvent(message_id=f"board-{e['id']}", activity_type='board_post', replace=True, timestamp=self._ms(ts),
                                             subagent_run_id=self.sub(e['agent']),
                                             content={'id': e['id'], 'agent': e['agent'], 'kind': e['kind'], 'content': e['content'], 'ts': ts}))
        elif kind == 'artifact':
            out.append(ActivitySnapshotEvent(message_id=f"artifact-{e['id']}", activity_type='artifact', replace=True, timestamp=self._ms(ts),
                                             subagent_run_id=self.sub(e['agent']),
                                             content={'id': e['id'], 'agent': e['agent'], 'name': e['name'], 'kind': e['kind'],
                                                      'content': e['content'], 'ts': ts}))
        elif kind == 'llm':
            out.append(ActivitySnapshotEvent(message_id=f"llm-{e['id']}", activity_type='llm_call', replace=True, timestamp=self._ms(ts),
                                             subagent_run_id=self.sub(e['agent']),
                                             content={k: e[k] for k in ('id', 'agent', 'step', 'purpose', 'model', 'prompt_tokens', 'completion_tokens',
                                                                        'total_tokens', 'latency_ms', 'status', 'error', 'estimated', 'in_chars', 'out_chars')} | {'ts': ts}))
        elif kind == 'access':
            out.append(CustomEvent(name='swarm.access', value={k: e[k] for k in ('reader', 'artifact_id', 'author', 'cross_agent')},
                                   subagent_run_id=self.sub(e['reader']), timestamp=self._ms(ts)))
            call = self.pending_read.pop(e['reader'], None)
            if call:
                art = self.ws.get_artifact(e['artifact_id'])
                body = f"artifact #{e['artifact_id']} by {e['author']}" + (f" — {art[3]}:\n{art[5]}" if art else '')
                out.append(self._tool_result(call, body, e['reader'], ts))
        elif kind == 'trace':
            out += self._map_trace(e)
        return out

    def _map_trace(self, e):
        out, ts, agent, ev, p = [], e['ts'], e['agent'], e['event'], e['payload']
        step = p.get('step')
        if agent == 'experiment':
            if ev == 'run_failed':
                self.crashed = p.get('error', 'experiment crashed')
            return out
        sid = self.sub(agent)
        if ev == 'run_started':
            self._ensure_started(agent, ts, out)
        elif ev == 'step_started':
            self._ensure_started(agent, ts, out)
            self._close_step(agent, ts, out)
            name = 'report' if step == 'report' else f'step-{step}'
            self.open_step[agent] = name
            out.append(StepStartedEvent(step_name=name, subagent_run_id=sid, timestamp=self._ms(ts)))
        elif ev == 'decision':
            self._ensure_started(agent, ts, out)
            mid = f"{agent}-s{step}-msg"
            conf = f" (confidence {p['confidence']:.2f})" if p.get('confidence') is not None else ''
            text = f"{p.get('action') or 'step'}{conf}" + (f" — {p['audit_summary']}" if p.get('audit_summary') else '')
            if p.get('rationale'):                  # the agent's written explanation of why it chose this step
                text += f"\nWhy: {p['rationale']}"
            out += list(self._text(mid, agent, text, ts))
            if p.get('data_query'):
                out += list(self._tool_call(f"{agent}-s{step}-data_query", 'data_query', p['data_query'], mid, agent, ts))
            if p.get('data_rows'):
                out += list(self._tool_call(f"{agent}-s{step}-data_rows", 'data_rows', p['data_rows'], mid, agent, ts))
            if p.get('artifact_to_read') is not None:
                cid = f"{agent}-s{step}-read_artifact"
                out += list(self._tool_call(cid, 'read_artifact', {'artifact_id': p['artifact_to_read']}, mid, agent, ts))
                self.pending_read[agent] = cid
        elif ev == 'query':
            art = self.ws.get_artifact(p.get('artifact_id'))
            out.append(self._tool_result(f"{agent}-s{step}-data_query", art[5] if art else 'query ran', agent, ts))
        elif ev == 'rows_viewed':
            body = (f"{p.get('count')} rows of '{p.get('table')}' returned to the agent "
                    f"(row ids {p.get('first')}–{p.get('last')}); recorded in the shared row coverage")
            out.append(self._tool_result(f"{agent}-s{step}-data_rows", body, agent, ts))
        elif ev == 'query_error':
            tool = p.get('tool') or 'data_query'
            out.append(self._tool_result(f"{agent}-s{step}-{tool}", f"ERROR: {p.get('error')}", agent, ts))
            out.append(CustomEvent(name='swarm.agent_error', value={'agent': agent, 'step': step, 'error': p.get('error'), 'kind': 'query'},
                                   subagent_run_id=sid, timestamp=self._ms(ts)))
        elif ev == 'error':
            self._ensure_started(agent, ts, out)
            self._close_step(agent, ts, out)
            out.append(CustomEvent(name='swarm.agent_error', value={'agent': agent, 'step': step, 'error': p.get('error'), 'kind': 'model'},
                                   subagent_run_id=sid, timestamp=self._ms(ts)))
            if agent == 'reporter':
                self.failed_agents.add(agent)
                out.append(SubagentErrorEvent(subagent_run_id=sid, message=p.get('error') or 'report failed', timestamp=self._ms(ts)))
        elif ev in ('report_progress', 'context_retry'):
            out.append(CustomEvent(name=f'swarm.{ev}', value={'agent': agent, **p}, subagent_run_id=sid, timestamp=self._ms(ts)))
        elif ev == 'cancelled':
            out.append(CustomEvent(name='swarm.agent_cancelled', value={'agent': agent, 'step': step}, subagent_run_id=sid, timestamp=self._ms(ts)))
        elif ev == 'run_failed':
            self._ensure_started(agent, ts, out)
            self._close_step(agent, ts, out)
            self.failed_agents.add(agent)
            out.append(SubagentErrorEvent(subagent_run_id=sid, message=p.get('error') or 'agent failed', timestamp=self._ms(ts)))
        elif ev == 'run_finished' and agent != 'reporter':
            self._close_step(agent, ts, out)
            self.finished_agents.add(agent)
            out.append(SubagentFinishedEvent(subagent_run_id=sid, result={'steps': p.get('steps'), 'done': p.get('done')},
                                             outcome={'type': 'success'}, timestamp=self._ms(ts)))
        return out
