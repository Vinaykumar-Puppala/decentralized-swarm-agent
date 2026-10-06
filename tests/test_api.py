"""FastAPI + AG-UI tests: real HTTP stack (Starlette TestClient), scripted fake model, the real sample Excel file."""
import json, time
from collections import Counter

import jsonpatch
import pytest
from fastapi.testclient import TestClient

import swarm.agent, swarm.experiment
from api.main import create_app
from test_experiment import ScriptedLLM


@pytest.fixture
def client(tmp_path, monkeypatch):
    fake = lambda cfg: ScriptedLLM()
    monkeypatch.setattr(swarm.agent, 'make_llm', fake)
    monkeypatch.setattr(swarm.experiment, 'make_llm', fake)
    ScriptedLLM.calls = {}
    with TestClient(create_app(tmp_path / 'api.sqlite')) as c:
        yield c


def upload(client, *paths):
    files = [('files', (name, data, 'application/octet-stream')) for name, data in paths]
    return client.post('/api/datasets', files=files)


def sample_xlsx():
    return ('Financial_Sample_Data.xlsx', open('Financial_Sample_Data.xlsx', 'rb').read())


def agui_body(objective='Explore this data and give deeper insights', run_id='r1', **forwarded):
    return {'threadId': 't1', 'runId': run_id, 'state': {}, 'tools': [], 'context': [],
            'messages': [{'id': 'u1', 'role': 'user', 'content': objective}], 'forwardedProps': forwarded}


def stream_events(client, body, limit=None):
    out = []
    with client.stream('POST', '/agui', json=body, headers={'accept': 'text/event-stream'}) as r:
        assert r.status_code == 200 and r.headers['content-type'].startswith('text/event-stream')
        for line in r.iter_lines():
            if line.startswith('data: '):
                out.append(json.loads(line[6:]))
                if limit and len(out) >= limit:
                    break
    return out


def check_protocol(events):
    """The AG-UI sequencing rules a client verifier enforces."""
    assert events[0]['type'] == 'RUN_STARTED'
    assert events[-1]['type'] in ('RUN_FINISHED', 'RUN_ERROR') and sum(e['type'] in ('RUN_FINISHED', 'RUN_ERROR') for e in events) == 1
    open_msgs, open_calls, open_steps, done_calls = set(), set(), set(), set()
    for e in events:
        t = e['type']
        if t == 'TEXT_MESSAGE_START':
            assert e['messageId'] not in open_msgs
            open_msgs.add(e['messageId'])
        elif t == 'TEXT_MESSAGE_CONTENT':
            assert e['messageId'] in open_msgs and e['delta']
        elif t == 'TEXT_MESSAGE_END':
            open_msgs.remove(e['messageId'])
        elif t == 'TOOL_CALL_START':
            open_calls.add(e['toolCallId'])
        elif t == 'TOOL_CALL_ARGS':
            assert e['toolCallId'] in open_calls
        elif t == 'TOOL_CALL_END':
            open_calls.remove(e['toolCallId']); done_calls.add(e['toolCallId'])
        elif t == 'TOOL_CALL_RESULT':
            assert e['toolCallId'] in done_calls
        elif t == 'STEP_STARTED':
            key = (e.get('subagentRunId'), e['stepName'])
            assert key not in open_steps
            open_steps.add(key)
        elif t == 'STEP_FINISHED':
            open_steps.remove((e.get('subagentRunId'), e['stepName']))
    assert not open_msgs and not open_calls and not open_steps


def final_state(events):
    state = None
    for e in events:
        if e['type'] == 'STATE_SNAPSHOT':
            state = e['snapshot']
        elif e['type'] == 'STATE_DELTA':
            state = jsonpatch.apply_patch(state, e['delta'])
    return state


def test_config_and_dataset_upload_without_joins(client):
    cfg = client.get('/api/config').json()
    assert cfg['providers'] == ['openai', 'anthropic', 'litellm', 'local'] and 'api_key' not in json.dumps(cfg)
    r = upload(client, sample_xlsx(), ('targets.csv', b'Country,Target\nFrance,1\nSpain,2\n'))
    assert r.status_code == 200
    d = r.json()
    assert [t['name'] for t in d['tables']] == ['Financial_Sample_Data', 'targets']
    assert d['tables'][0]['rows'] == 704 and 'SEPARATE TABLES' in d['profile']
    page = client.get(f"/api/datasets/{d['dataset_id']}/rows", params={'table': 'Financial_Sample_Data', 'offset': 700, 'limit': 10}).json()
    assert page['total'] == 704 and page['row_ids'] == [700, 701, 702, 703] and page['columns'][0] == 'Segment'
    assert client.get(f"/api/datasets/{d['dataset_id']}/rows", params={'table': 'nope'}).status_code == 404
    assert upload(client, ('x.txt', b'hi')).status_code == 400
    assert client.get('/api/datasets/zzz').status_code == 404


def test_llm_test_endpoint_reports_missing_model(client):
    r = client.post('/api/llm/test', json={'provider': 'local', 'model': ''}).json()
    assert r['ok'] is False and 'MODEL' in r['error']


def test_agui_full_run_is_protocol_correct_and_state_reconstructs(client):
    did = upload(client, sample_xlsx()).json()['dataset_id']
    events = stream_events(client, agui_body(llm={'provider': 'local', 'model': 'fake'}, nAgents=3, steps=3, datasetId=did))
    check_protocol(events)
    types = Counter(e['type'] for e in events)
    assert events[0]['threadId'] == 't1' and events[0]['runId'] == 'r1' and events[1]['type'] == 'STATE_SNAPSHOT'
    assert events[-1]['type'] == 'RUN_FINISHED' and events[-1]['outcome']['type'] == 'success'
    assert events[-1]['result']['report'].startswith('# Report')

    started = [e for e in events if e['type'] == 'SUBAGENT_STARTED']
    assert sorted(e['name'] for e in started) == ['agent-1', 'agent-2', 'agent-3', 'reporter']
    assert types['SUBAGENT_FINISHED'] + types['SUBAGENT_ERROR'] == 4                # every sub-agent is closed
    assert all(e['subagentRunId'].split(':')[1] in ('agent-1', 'agent-2', 'agent-3', 'reporter')
               for e in events if 'subagentRunId' in e)

    calls = Counter(e['toolCallName'] for e in events if e['type'] == 'TOOL_CALL_START')
    assert calls['data_query'] >= 2 and calls['data_rows'] >= 2 and calls['read_artifact'] >= 1
    results = [e for e in events if e['type'] == 'TOOL_CALL_RESULT']
    assert any('Government' in e['content'] for e in results if e['toolCallId'].endswith('data_query'))
    assert any('rows of' in e['content'] for e in results if e['toolCallId'].endswith('data_rows'))

    acts = Counter(e['activityType'] for e in events if e['type'] == 'ACTIVITY_SNAPSHOT')
    assert acts['board_post'] >= 2 and acts['artifact'] >= 4
    customs = Counter(e['name'] for e in events if e['type'] == 'CUSTOM')
    assert customs['swarm.agent_error'] >= 1                                        # agent-3's simulated timeout
    assert types['STATE_DELTA'] >= 3 and types['STEP_STARTED'] >= 6

    # The shared state rebuilt on the client from snapshot + JSON-Patch deltas equals the server's own view.
    state = final_state(events)
    summary = client.get(f"/api/runs/{state['swarmRunId']}").json()
    assert state['status'] == summary['status'] == 'finished'
    assert state['counts'] == summary['counts']
    assert {a: v['steps'] for a, v in state['agents'].items()} == {a['agent']: a['steps'] for a in summary['agents']}
    cov = state['coverage']['Financial_Sample_Data']
    assert cov['total'] == 704 and cov['read'] == sum(b - a + 1 for a, b in cov['ranges']) and cov['read'] > 0
    assert 'api_key' not in json.dumps(state)


def test_agui_reattach_replays_a_finished_run(client):
    first = stream_events(client, agui_body(llm={'provider': 'local', 'model': 'fake'}, nAgents=2, steps=2, report=False))
    swarm_run = final_state(first)['swarmRunId']
    again = stream_events(client, agui_body(run_id='r2', attachRunId=swarm_run))
    check_protocol(again)
    assert again[0]['runId'] == 'r2' and final_state(again)['swarmRunId'] == swarm_run
    assert final_state(again)['counts'] == final_state(first)['counts']
    assert Counter(e['type'] for e in again if e['type'].startswith('SUBAGENT')) == Counter(e['type'] for e in first if e['type'].startswith('SUBAGENT'))


def test_agui_errors_are_in_stream(client):
    ev = stream_events(client, agui_body(llm={'provider': 'local', 'model': ''}))
    assert [e['type'] for e in ev] == ['RUN_STARTED', 'RUN_ERROR'] and 'MODEL' in ev[1]['message'] and ev[1]['code'] == 'http_400'
    ev = stream_events(client, agui_body(attachRunId='nope'))
    assert ev[-1]['type'] == 'RUN_ERROR' and ev[-1]['code'] == 'http_404'
    ev = stream_events(client, agui_body(llm={'provider': 'local', 'model': 'm'}, datasetId='missing'))
    assert ev[-1]['type'] == 'RUN_ERROR'
    assert client.post('/agui', json={'bad': 1}).status_code == 422


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A real uvicorn server on a free port (TestClient buffers whole responses, so it cannot model a client hanging up)."""
    import socket, threading, uvicorn

    class Slower(ScriptedLLM):
        def invoke(self, msgs):
            time.sleep(1.0)
            return super().invoke(msgs)
    slow = lambda cfg: Slower()
    monkeypatch.setattr(swarm.agent, 'make_llm', slow)
    monkeypatch.setattr(swarm.experiment, 'make_llm', slow)
    ScriptedLLM.calls = {}
    sock = socket.socket(); sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]; sock.close()
    server = uvicorn.Server(uvicorn.Config(create_app(tmp_path / 'live.sqlite'), host='127.0.0.1', port=port, log_level='warning'))
    t = threading.Thread(target=server.run, daemon=True); t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f'http://127.0.0.1:{port}'
    server.should_exit = True
    t.join(10)


def test_dropped_connection_does_not_kill_the_run_and_stop_cancels(live):
    import httpx
    api = httpx.Client(base_url=live, timeout=30)
    did = api.post('/api/datasets', files=[('files', ('Financial_Sample_Data.xlsx', sample_xlsx()[1]))]).json()['dataset_id']
    body = agui_body(llm={'provider': 'local', 'model': 'fake'}, nAgents=2, steps=8, datasetId=did, report=True)
    head = []
    with api.stream('POST', '/agui', json=body) as r:                  # read a few events, then hang up for real
        for line in r.iter_lines():
            if line.startswith('data: '):
                head.append(json.loads(line[6:]))
            if len(head) >= 4:
                break
    run_id = final_state(head)['swarmRunId']
    time.sleep(0.8)
    assert api.get(f'/api/runs/{run_id}').json()['status'] == 'running'             # still going after the disconnect

    busy = stream_events(api, agui_body(run_id='r9', llm={'provider': 'local', 'model': 'fake'}))
    assert busy[-1]['type'] == 'RUN_ERROR' and busy[-1]['code'] == 'http_409'       # one active run at a time
    assert [r['active'] for r in api.get('/api/runs').json()] == [True]

    assert api.post(f'/api/runs/{run_id}/stop').json()['status'] == 'cancelling'
    for _ in range(100):
        if api.get(f'/api/runs/{run_id}').json()['status'] == 'cancelled':
            break
        time.sleep(0.2)
    assert api.get(f'/api/runs/{run_id}').json()['status'] == 'cancelled'
    replay = stream_events(api, agui_body(run_id='r10', attachRunId=run_id))
    check_protocol(replay)
    assert replay[-1]['type'] == 'RUN_FINISHED' and replay[-1]['outcome']['type'] == 'cancelled'
    assert api.post(f'/api/runs/{run_id}/stop').status_code == 409
    assert api.get('/api/runs/unknown').status_code == 404
