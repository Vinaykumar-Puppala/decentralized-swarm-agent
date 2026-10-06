"""Translator-level tests: deterministic workspace rows in, AG-UI events out."""
import json
from swarm.agui import AguiTranslator
from swarm.workspace import Workspace


def types(evs):
    return [e.type.value for e in evs]


def test_read_artifact_access_becomes_tool_result_and_custom_event(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('obj', {'tables': {'t': 10}, 'n_agents': 2})
    tr = AguiTranslator(ws, 'th', 'rn')
    begin = tr.begin()
    assert types(begin) == ['RUN_STARTED', 'STATE_SNAPSHOT']

    aid = ws.artifact('agent-2', 'finding', 'revenue is up 5%')
    ws.trace('agent-1', 'run_started', {'max_steps': 2})
    ws.trace('agent-1', 'step_started', {'step': 1})
    ws.trace('agent-1', 'decision', {'step': 1, 'action': 'read peer', 'audit_summary': 'checking', 'confidence': 0.5,
                                     'artifact_to_read': aid, 'data_rows': {'table': 't', 'limit': 3}})
    assert ws.access('agent-1', aid) is not None
    ws.log_rows('agent-1', 't', [0, 1, 2])
    ws.trace('agent-1', 'rows_viewed', {'step': 1, 'table': 't', 'count': 3, 'first': 0, 'last': 2})
    ws.trace('agent-1', 'query_error', {'step': 1, 'tool': 'data_query', 'query': {'x': 1}, 'error': 'Unknown column'})
    evs = tr.poll()
    names = [(e.type.value, getattr(e, 'tool_call_name', None) or getattr(e, 'name', None) or getattr(e, 'activity_type', None)) for e in evs]
    assert ('TOOL_CALL_START', 'read_artifact') in names and ('TOOL_CALL_START', 'data_rows') in names
    assert ('CUSTOM', 'swarm.access') in names and ('CUSTOM', 'swarm.agent_error') in names
    assert ('ACTIVITY_SNAPSHOT', 'artifact') in names

    res = {e.tool_call_id: e.content for e in evs if e.type.value == 'TOOL_CALL_RESULT'}
    assert 'revenue is up 5%' in res['agent-1-s1-read_artifact'] and 'by agent-2' in res['agent-1-s1-read_artifact']
    assert '3 rows' in res['agent-1-s1-data_rows']
    assert res['agent-1-s1-data_query'].startswith('ERROR: Unknown column')
    # text message carries the audit summary, attributed to the agent's sub-run
    msg = next(e for e in evs if e.type.value == 'TEXT_MESSAGE_CONTENT')
    assert msg.delta == 'read peer (confidence 0.50) — checking' and msg.subagent_run_id.endswith(':agent-1')
    # coverage landed in the shared state delta
    delta = [e for e in evs if e.type.value == 'STATE_DELTA'][-1].delta
    assert any(op.path.startswith('/coverage/t') for op in delta) and any('/agents/agent-1' in op.path for op in delta)


def test_failed_and_interrupted_runs(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('obj', {})
    tr = AguiTranslator(ws, 'th', 'rn'); tr.begin()
    ws.trace('agent-1', 'run_started', {}); ws.trace('agent-1', 'step_started', {'step': 1})
    ws.trace('agent-1', 'run_failed', {'error': 'boom'})
    ws.trace('experiment', 'run_failed', {'error': 'executor died'})
    ws.finish_run('crashed')
    evs = tr.poll()
    assert 'SUBAGENT_ERROR' in types(evs) and types(evs)[-1] == 'RUN_ERROR' and evs[-1].message == 'executor died'
    assert types(evs).count('STEP_FINISHED') == 1 and tr.done

    ws2 = Workspace(tmp_path / 'w2.sqlite').start_run('obj', {})
    tr2 = AguiTranslator(ws2, 'th', 'rn'); tr2.begin()
    ws2.trace('agent-1', 'run_started', {}); ws2.finish_run('interrupted')
    out = tr2.poll()
    assert 'SUBAGENT_ERROR' in types(out) and types(out)[-1] == 'RUN_ERROR'


def test_report_is_streamed_in_chunks(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('obj', {})
    tr = AguiTranslator(ws, 'th', 'rn'); tr.begin()
    ws.trace('agent-1', 'run_started', {}); ws.trace('agent-1', 'run_finished', {'steps': 1, 'done': True})
    ws.trace('reporter', 'step_started', {'step': 'report'}); ws.trace('reporter', 'run_finished', {'steps': 'report', 'done': True})
    report = 'x' * 400
    ws.finish_run('finished', report)
    evs = tr.poll()
    chunks = [e.delta for e in evs if e.type.value == 'TEXT_MESSAGE_CONTENT']
    assert len(chunks) == 3 and ''.join(chunks) == report
    assert types(evs)[-1] == 'RUN_FINISHED' and evs[-1].result['report'] == report
    assert types(evs).count('SUBAGENT_FINISHED') == 2
