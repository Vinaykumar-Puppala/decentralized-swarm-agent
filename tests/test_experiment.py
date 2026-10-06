"""End-to-end swarm run on the real sample file with a scripted fake model (no network)."""
import json, re, threading
from langchain_core.messages import AIMessage
import swarm.agent, swarm.experiment
from swarm.data import Dataset
from swarm.experiment import run
from swarm.llm import LLMConfig
from swarm.workspace import Workspace


class ScriptedLLM:
    """Behaves like a chat model. Per agent: step1 query, step2 read someone else's artifact (or garbage reply on agent-2),
    step3 publish final answer. agent-3 raises on its first call to exercise error handling."""
    lock = threading.Lock()
    calls = {}

    def invoke(self, msgs):
        text = msgs[-1].content
        if text.startswith('Your reply could not be parsed'):
            return AIMessage(content='{"status": "continue", "action": "repaired", "audit_summary": "fixed"}')
        if text.startswith('OBJECTIVE:') and 'BOARD:' in text and 'AGENT ID' not in text:
            return AIMessage(content='# Report\nAll good.')
        agent = re.search(r'AGENT ID: (\S+)', text).group(1)
        step = int(re.search(r'STEP: (\d+)', text).group(1))
        with self.lock:
            n = self.calls[agent] = self.calls.get(agent, 0) + 1
        if agent == 'agent-3' and n == 1:
            raise TimeoutError('simulated provider timeout')
        if step == 1 and 'FINAL STEP' not in text:
            return AIMessage(content=json.dumps({'status': 'continue', 'action': 'profit by segment', 'audit_summary': 's',
                                                 'data_query': {'group_by': ['Segment'], 'metrics': [{'column': 'Profit', 'agg': 'sum'}],
                                                                'sort_by': 'Profit_sum'},
                                                 'board_message': f'{agent} looking at segments'}))
        if step == 2 and 'FINAL STEP' not in text:
            assert 'RESULTS FROM YOUR LAST STEP' in text and 'Government' in text   # query result was fed back
            if agent == 'agent-2':
                return AIMessage(content='I think I will read something.')       # unparseable -> repair path
            others = [int(i) for i, a in re.findall(r'^(\d+) \| (agent-\d+) \|', text, re.M) if a != agent]
            return AIMessage(content='```json\n' + json.dumps({'status': 'continue', 'action': 'read peer', 'audit_summary': 's',
                                                                'artifact_to_read': others[0] if others else 9999}) + '\n```')
        return AIMessage(content=json.dumps({'status': 'done', 'action': 'final', 'audit_summary': 's', 'confidence': 0.8,
                                             'artifact_name': 'final', 'artifact_content': ['Government leads profit']}))


def test_end_to_end_with_sample_file(tmp_path, monkeypatch):
    fake = lambda cfg: ScriptedLLM()
    monkeypatch.setattr(swarm.agent, 'make_llm', fake)
    monkeypatch.setattr(swarm.experiment, 'make_llm', fake)
    ScriptedLLM.calls = {}
    db = tmp_path / 'ws.sqlite'
    ds = Dataset.from_source('Financial_Sample_Data.xlsx')
    out = run('Explore this data and give deeper insights', LLMConfig('local', 'fake'), 5, 3, db, ds, max_start_jitter=0.2)

    assert out['final_report'].startswith('# Report')
    assert all(not r.get('error') for r in out['results'])
    ws = Workspace(db).for_run(out['run_id'])
    s = ws.snapshot()
    kinds = [a[4] for a in s['artifacts']]
    events = [t[3] for t in s['traces']]
    assert kinds.count('query_result') >= 4 and 'Government' in s['artifacts'][0][5]
    assert kinds.count('final_answer') >= 3
    assert 'error' in events and events.count('run_finished') == 5                  # agent-3 failure was contained
    assert any(json.loads(t[4]).get('action') == 'repaired' for t in s['traces'] if t[3] == 'decision')
    assert any(a[5] == 1 for a in s['accesses'])                                     # real cross-agent read
    assert ws.run_info()[4] == 'finished' and '"api_key": ""' in ws.run_info()[3]

    # a second run starts from a clean workspace
    out2 = run('second objective', LLMConfig('local', 'fake'), 1, 1, db, None, do_synthesis=False)
    assert ws.for_run(out2['run_id']).snapshot()['artifacts'][0][4] == 'final_answer'
    assert len(ws.for_run(out2['run_id']).snapshot()['artifacts']) == 1
