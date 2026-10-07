"""Shared activity log + personal scratchpads: logged with a written rationale, readable by peers only if they choose to."""
import json, re
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import swarm.agent
from api.main import create_app
from swarm.agent import run_agent
from swarm.agui import AguiTranslator
from swarm.data import Dataset
from swarm.llm import LLMConfig
from swarm.workspace import Workspace

CFG = LLMConfig('local', 'fake')


class PlanLLM:
    """Replies with a fixed plan per agent (one reply per step) and records every prompt it was sent."""
    plans, prompts = {}, {}

    def invoke(self, msgs):
        user = msgs[-1].content
        agent = re.search(r'AGENT ID: (\S+)', user).group(1)
        step = int(re.search(r'STEP: (\d+)', user).group(1))
        self.prompts.setdefault(agent, {})[step] = (msgs[0].content, user)
        return AIMessage(content=json.dumps(self.plans[agent][step - 1]))


@pytest.fixture
def env(tmp_path, monkeypatch):
    PlanLLM.plans, PlanLLM.prompts = {}, {}
    monkeypatch.setattr(swarm.agent, 'make_llm', lambda cfg: PlanLLM())
    ws = Workspace(tmp_path / 'w.sqlite').start_run('obj')
    ds = Dataset.from_source('Financial_Sample_Data.xlsx')
    return ws, ds


def go(ws, ds, agent, plan, visibility='full'):
    PlanLLM.plans[agent] = plan
    return run_agent(agent, 'obj', ws, CFG, len(plan), ds, visibility=visibility)


RATIONALE_1 = 'Segments differ most in margin, so I group profit by segment first; alternatives were country or product but those split the data thinner.'
NOTE_1 = 'hypothesis: Enterprise loses money because of discounts'
STEP_1 = {'status': 'continue', 'action': 'profit by segment', 'audit_summary': 'grouping profit', 'rationale': RATIONALE_1, 'confidence': 0.7,
          'data_query': {'group_by': ['Segment'], 'metrics': [{'column': 'Profit', 'agg': 'sum'}]},
          'scratchpad_note': NOTE_1, 'board_message': 'looking at segment profit'}
DONE = {'status': 'done', 'action': 'final', 'audit_summary': 'done', 'rationale': 'wrapping up', 'artifact_name': 'a', 'artifact_content': 'b'}


def test_each_step_is_logged_with_rationale_tools_and_outcomes(env):
    ws, ds = env
    go(ws, ds, 'agent-1', [STEP_1, DONE])
    steps = ws.activity_entries(kinds=('step',), limit=10)
    assert [(r[2], r[3], r[5]) for r in steps] == [('agent-1', 1, 'profit by segment'), ('agent-1', 2, 'final')]
    first = steps[0]
    assert first[7] == RATIONALE_1 and first[8] == 0.7                        # the detailed "why", and confidence
    tools = {t['tool']: t for t in json.loads(first[9])}
    assert set(tools) == {'data_query', 'scratchpad_note', 'board_message'}
    assert tools['data_query']['ok'] and tools['data_query']['outcome'].startswith('Query result')
    assert json.loads(steps[1][9])[0]['tool'] == 'publish_artifact'
    # own notes persist and come back in the agent's own next prompt
    assert [n[3] for n in ws.scratch_notes('agent-1')] == [NOTE_1]
    assert NOTE_1 in PlanLLM.prompts['agent-1'][2][1] and 'YOUR SCRATCHPAD' in PlanLLM.prompts['agent-1'][2][1]


def test_peers_are_never_shown_anything_unless_they_ask(env):
    ws, ds = env
    go(ws, ds, 'agent-1', [STEP_1, DONE])
    # agent-2 never asks: it sees only a pointer that the log exists, none of agent-1's content, and no read is recorded
    go(ws, ds, 'agent-2', [{**DONE, 'rationale': 'independent'}])
    system, user = PlanLLM.prompts['agent-2'][1]
    assert 'SHARED ACTIVITY LOG: 2 step entries so far (agent-1: 2)' in user and 'You are free to ignore it' in user
    assert "OTHER AGENTS' SCRATCHPADS" in user and 'you never have to' in user
    assert RATIONALE_1 not in user and NOTE_1 not in user and 'grouping profit' not in user
    assert ws.activity_entries(kinds=('read',), limit=10) == []
    assert 'Entirely optional' in system and 'never expected to read it' in system


def test_a_read_is_chosen_delivered_next_step_and_logged_as_its_own_entry(env):
    ws, ds = env
    go(ws, ds, 'agent-1', [STEP_1, DONE])
    go(ws, ds, 'agent-3', [
        {'status': 'continue', 'action': 'see what others tried', 'audit_summary': 'checking peers', 'rationale': 'avoid repeating agent-1',
         'read_log': {'agent': 'agent-1', 'limit': 5}, 'read_scratchpad': 'agent-1'},
        DONE])
    inbox = PlanLLM.prompts['agent-3'][2][1]
    assert 'ACTIVITY LOG of agent-1 (2 entries' in inbox and RATIONALE_1 in inbox and 'tried: data_query' in inbox
    assert 'SCRATCHPAD of agent-1 (1 notes)' in inbox and NOTE_1 in inbox
    reads = ws.activity_entries(kinds=('read',), limit=10)
    assert [(r[2], r[3], r[10]) for r in reads] == [('agent-3', 1, 'log:agent-1'), ('agent-3', 1, 'scratchpad:agent-1')]
    step = [r for r in ws.activity_entries(agent='agent-3', kinds=('step',), limit=5)][0]
    assert {t['tool'] for t in json.loads(step[9])} == {'read_log', 'read_scratchpad'}
    traces = {t[3] for t in ws.snapshot()['traces']}
    assert {'activity_read', 'scratchpad_read'} <= traces


def test_visibility_levels(env):
    ws, ds = env
    ask = [{'status': 'continue', 'action': 'peek', 'audit_summary': 's', 'rationale': 'r', 'read_log': True, 'read_scratchpad': 'agent-1'}, DONE]
    go(ws, ds, 'agent-1', [STEP_1, DONE])

    go(ws, ds, 'agent-2', ask, visibility='isolated')                       # nothing of peers is readable or even advertised
    system, user = PlanLLM.prompts['agent-2'][1]
    assert 'read_log' not in system and 'read_scratchpad' not in system and 'SHARED ACTIVITY LOG' not in user
    assert 'not available in this run' in PlanLLM.prompts['agent-2'][2][1] and "private in this run" in PlanLLM.prompts['agent-2'][2][1]

    go(ws, ds, 'agent-3', ask, visibility='log')                            # log yes, other agents' scratchpads no
    system, user = PlanLLM.prompts['agent-3'][1]
    assert 'read_log' in system and 'read_scratchpad' not in system and "OTHER AGENTS' SCRATCHPADS" not in user
    inbox = PlanLLM.prompts['agent-3'][2][1]
    assert RATIONALE_1 in inbox and NOTE_1 not in inbox and "private in this run" in inbox

    reads = [(r[2], r[10]) for r in ws.activity_entries(kinds=('read',), limit=10)]
    assert reads == [('agent-3', 'log:all')]                                # denied reads leave no 'read' entry


def test_denied_and_failed_tools_show_up_as_failures_in_the_log(env):
    ws, ds = env
    go(ws, ds, 'agent-1', [{'status': 'continue', 'action': 'bad query', 'audit_summary': 's', 'rationale': 'try a missing column',
                            'data_query': {'group_by': ['NoSuchColumn'], 'metrics': [{'column': 'Profit', 'agg': 'sum'}]}}, DONE])
    entry = ws.activity_entries(agent='agent-1', kinds=('step',), limit=1)[0]
    # (limit=1 returns the newest, the final step; look at the first step explicitly)
    first = ws.activity_entries(agent='agent-1', kinds=('step',), limit=5)[0]
    t = json.loads(first[9])[0]
    assert t['tool'] == 'data_query' and t['ok'] is False and 'Unknown column' in t['outcome'] and entry[3] == 2


def test_model_error_is_logged_for_others_to_learn_from(env, monkeypatch):
    ws, ds = env

    class Boom:
        def invoke(self, msgs):
            raise TimeoutError('provider timed out')
    monkeypatch.setattr(swarm.agent, 'make_llm', lambda cfg: Boom())
    monkeypatch.setattr(swarm.agent.time, 'sleep', lambda s: None)
    run_agent('agent-1', 'obj', ws, CFG, 2, ds)
    errs = ws.activity_entries(kinds=('error',), limit=10)
    assert errs and 'TimeoutError' in errs[0][6]


def test_visibility_is_validated_end_to_end(tmp_path):
    with pytest.raises(ValueError, match='visibility'):
        swarm.agent.Agent('agent-1', Workspace(tmp_path / 'x.sqlite').start_run('o'), CFG, None, visibility='everything')
    with TestClient(create_app(tmp_path / 'api.sqlite')) as c:
        r = c.post('/api/runs', json={'objective': 'x', 'llm': {'provider': 'local', 'model': 'm'}, 'visibility': 'bogus'})
        assert r.status_code == 400 and 'visibility' in r.json()['detail']


def test_rationale_reaches_the_ag_ui_feed(tmp_path):
    ws = Workspace(tmp_path / 'w.sqlite').start_run('obj', {})
    tr = AguiTranslator(ws, 't', 'r'); tr.begin()
    ws.trace('agent-1', 'run_started', {}); ws.trace('agent-1', 'step_started', {'step': 1})
    ws.trace('agent-1', 'decision', {'step': 1, 'action': 'look', 'audit_summary': 'sum', 'rationale': 'because X beats Y'})
    text = ''.join(e.delta for e in tr.poll() if e.type.value == 'TEXT_MESSAGE_CONTENT')
    assert text == 'look — sum' + chr(10) + 'Why: because X beats Y'
