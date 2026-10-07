"""Model-call logging, token accounting and context-limit handling (no network)."""
import json
from langchain_core.messages import AIMessage
import swarm.agent, swarm.experiment
from swarm.experiment import synthesize
from swarm.llm import LLMConfig, TrackedLLM, is_context_error
from swarm.workspace import Workspace
from swarm.summary import RunView

CFG = LLMConfig('local', 'fake')
DONE = json.dumps({'status': 'done', 'action': 'finish', 'audit_summary': 'ok', 'artifact_name': 'a', 'artifact_content': 'result'})


class Reporting:
    """Reports provider token usage like langchain's usage_metadata."""
    def invoke(self, msgs):
        return AIMessage(content=DONE, usage_metadata={'input_tokens': 100, 'output_tokens': 20, 'total_tokens': 120})


class NoUsage:
    def invoke(self, msgs):
        return AIMessage(content='x' * 40)


def test_every_call_is_logged_with_provider_tokens(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    out = TrackedLLM(Reporting(), CFG, ws, 'agent-1').call([swarm.agent.SystemMessage(content='sys'), swarm.agent.HumanMessage(content='hi')], 'decision', 3)
    assert out == DONE
    c = ws.llm_call(1)
    assert [m['role'] for m in c['input']] == ['system', 'user'] and c['input'][1]['content'] == 'hi' and c['output'] == DONE
    assert (c['agent'], c['step'], c['purpose'], c['prompt_tokens'], c['completion_tokens'], c['total_tokens'], c['estimated']) == ('agent-1', 3, 'decision', 100, 20, 120, False)
    assert ws.usage_summary()['total']['total'] == 120


def test_tokens_are_estimated_when_the_provider_reports_none(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    TrackedLLM(NoUsage(), CFG, ws, 'a').call([swarm.agent.HumanMessage(content='y' * 80)], 'decision')
    c = ws.llm_call(1)
    assert c['estimated'] and c['prompt_tokens'] == 20 and c['completion_tokens'] == 10
    assert ws.usage_summary()['total']['estimated']


def test_failed_calls_are_logged_and_reraised(tmp_path):
    class Boom:
        def invoke(self, msgs): raise TimeoutError('slow')
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    try:
        TrackedLLM(Boom(), CFG, ws, 'a').call([swarm.agent.HumanMessage(content='q')], 'decision')
        assert False
    except TimeoutError:
        pass
    c = ws.llm_call(1)
    assert c['status'] == 'error' and 'slow' in c['error']
    assert ws.usage_summary()['total']['errors'] == 1


def test_runview_totals_tokens_per_agent_and_overall(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    for agent in ('agent-1', 'agent-1', 'agent-2'):
        TrackedLLM(Reporting(), CFG, ws, agent).call([swarm.agent.HumanMessage(content='q')], 'decision', 1)
    v = RunView()
    evs = v.update(ws)
    assert [e['type'] for e in evs] == ['llm'] * 3 and evs[0]['total_tokens'] == 120
    t = v.token_summary()
    assert t['total']['total'] == 360 and t['agents']['agent-1']['calls'] == 2 and t['agents']['agent-2']['prompt'] == 100


def test_context_error_detection():
    assert is_context_error(ValueError("This model's maximum context length is 8192 tokens"))
    assert is_context_error(RuntimeError('prompt is too long: 250000 tokens'))
    assert not is_context_error(TimeoutError('timed out'))


class Limited:
    """Rejects any prompt over `limit` characters with a context error; condenses parts; writes the report."""
    def __init__(self, limit): self.limit, self.seen = limit, []

    def invoke(self, msgs):
        total = sum(len(m.content) for m in msgs)
        self.seen.append((msgs[0].content[:20], total))
        if total > self.limit:
            raise ValueError("This model's maximum context length is exceeded")
        return AIMessage(content='# Report' if msgs[0].content.startswith('You are a post-hoc') else '- condensed')


def test_reporter_condenses_oversized_material_in_pieces(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    for i in range(12):
        ws.artifact(f'agent-{i % 3 + 1}', f'finding {i}', 'data ' * 400, kind='finding')      # ~24k characters in total
    fake = Limited(limit=7000)
    monkeypatch.setattr(swarm.experiment, 'make_llm', lambda cfg: fake)
    assert synthesize(ws, 'obj', CFG, chunk_chars=5000) == '# Report'
    purposes = [r[0] for r in ws._q("SELECT purpose FROM llm_calls WHERE run_id=? ORDER BY id", (ws.run_id,))]
    assert purposes.count('report-map') >= 2 and purposes[-1] == 'report'
    assert all(t <= 7000 for _, t in fake.seen)


def test_reporter_halves_the_chunk_size_after_a_context_error(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    for i in range(6):
        ws.artifact('agent-1', f'f{i}', 'word ' * 600, kind='finding')
    fake = Limited(limit=4500)               # the configured chunk (20k) is too big for this model
    monkeypatch.setattr(swarm.experiment, 'make_llm', lambda cfg: fake)
    assert synthesize(ws, 'obj', CFG, chunk_chars=20000) == '# Report'
    assert 'context_retry' in [t[3] for t in ws.snapshot()['traces']]


def test_agent_retries_once_with_a_shorter_prompt_on_context_error(tmp_path, monkeypatch):
    prompts = []

    class Tight:
        def invoke(self, msgs):
            prompts.append(len(msgs[-1].content))
            if len(prompts) == 1:
                raise ValueError('context_length_exceeded')
            return AIMessage(content=DONE)

    monkeypatch.setattr(swarm.agent, 'make_llm', lambda cfg: Tight())
    ws = Workspace(tmp_path / 'w.sqlite').start_run('o')
    for i in range(20):
        ws.board('agent-9', 'note', f'message {i} ' + 'x' * 200)
    out = swarm.agent.run_agent('agent-1', 'obj', ws, CFG, steps=2)
    assert out['done'] and len(prompts) == 2 and prompts[1] < prompts[0]
    assert 'context_retry' in [t[3] for t in ws.snapshot()['traces']]
    assert [r[0] for r in ws._q("SELECT status FROM llm_calls ORDER BY id")] == ['error', 'ok']
