"""Are the five agents really identical and autonomous? Capture the exact prompt each one is sent on step 1."""
import difflib, re, threading
from concurrent.futures import ThreadPoolExecutor

from langchain_core.messages import AIMessage

import swarm.agent
from swarm.agent import run_agent
from swarm.data import Dataset
from swarm.llm import LLMConfig
from swarm.workspace import Workspace

OBJECTIVE = 'Explore this data and give deeper insights.'
N = 5


class RecordingLLM:
    """Stands in for the model. Records what each agent sends; a barrier makes all five finish building their
    prompt before any of them acts, so every agent sees the same (empty) shared workspace."""
    barrier = threading.Barrier(N)
    seen, lock = {}, threading.Lock()

    def invoke(self, msgs):
        system, user = msgs[0].content, msgs[-1].content
        agent = re.search(r'AGENT ID: (\S+)', user).group(1)
        with self.lock:
            self.seen[agent] = (system, user)
        self.barrier.wait(timeout=20)
        return AIMessage(content='{"status": "done", "action": "noop", "audit_summary": "recorded"}')


def capture(tmp_path, monkeypatch):
    RecordingLLM.seen, RecordingLLM.barrier = {}, threading.Barrier(N)
    monkeypatch.setattr(swarm.agent, 'make_llm', lambda cfg: RecordingLLM())
    ws = Workspace(tmp_path / 'w.sqlite').start_run(OBJECTIVE)
    ds = Dataset.from_source('Financial_Sample_Data.xlsx')
    cfg = LLMConfig('local', 'fake')
    with ThreadPoolExecutor(N) as ex:
        list(ex.map(lambda i: run_agent(f'agent-{i + 1}', OBJECTIVE, ws, cfg, 1, ds), range(N)))
    return RecordingLLM.seen


def test_five_agents_get_identical_prompts_apart_from_their_id(tmp_path, monkeypatch, capsys):
    seen = capture(tmp_path, monkeypatch)
    assert sorted(seen) == [f'agent-{i}' for i in range(1, N + 1)]

    systems = {a: s for a, (s, _) in seen.items()}
    assert len(set(systems.values())) == 1                                   # identical system prompt, byte for byte

    mask = lambda a, u: u.replace(f'AGENT ID: {a}', 'AGENT ID: <id>')
    users = {a: mask(a, u) for a, (_, u) in seen.items()}
    assert len(set(users.values())) == 1                                     # identical user prompt once the id is masked

    # the ONLY line that differs between two agents' raw prompts is the id line
    a1, a2 = seen['agent-1'][1].splitlines(), seen['agent-2'][1].splitlines()
    changed = [l for l in difflib.unified_diff(a1, a2, lineterm='', n=0) if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    assert changed == ['-AGENT ID: agent-1   STEP: 1 of 1', '+AGENT ID: agent-2   STEP: 1 of 1']

    system, user = seen['agent-1']
    # autonomy: no planner, no roles, no peer chat, and nothing about reading peers' work is mandatory
    assert 'NO coordinator, planner, leader, assigned role, or direct agent-to-agent chat' in system
    assert 'Using it is optional' in system
    assert not re.search(r'\b(you must|you have to|always|required to|be sure to)\b[^.\n]{0,40}\b(read|check|consult|review)\b', system + user, re.I)
    assert not re.search(r'agent-[1-5]\b', user.replace('AGENT ID: agent-1', ''))   # no other agent is named: no roles, no hints

    with capsys.disabled():
        shown = re.sub(r'(DATA: .*?\):\n).*?(\n\ndata_query format)', r'\1<... dataset profile, identical for every agent ...>\2', user, flags=re.S)
        print('\n' + '=' * 78 + '\nSYSTEM PROMPT (sent to all 5 agents)\n' + '=' * 78 + '\n' + system)
        print('\n' + '=' * 78 + '\nUSER PROMPT AS RECEIVED BY agent-1 (dataset profile elided for length)\n' + '=' * 78 + '\n' + shown)
        print('\n' + '=' * 78 + f"\nsystem prompts identical: {len(set(systems.values())) == 1}   user prompts identical (id masked): "
              f"{len(set(users.values())) == 1}\nsystem {len(system):,} chars, user {len(user):,} chars\n"
              f"the one differing line: {changed}\n" + '=' * 78)
