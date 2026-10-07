import random, threading
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage
from .agent import VISIBILITY, run_agent
from .llm import LLMConfig, make_llm, response_text
from .workspace import Workspace

REPORTER = '''You are a post-hoc reporter. You did NOT take part in the experiment and must not add new analysis or numbers.
Summarise what the agents found into a clear report for a business reader: key insights (with the agents' computed numbers),
data-quality caveats, points where agents disagreed, and open questions. Attribute claims to agent IDs. Use Markdown.'''


def synthesize(ws, objective, llm_cfg, budget=40000):
    """Observer-only summary of a finished run. It reads the workspace; agents never see its output."""
    snap = ws.snapshot()
    order = {'final_answer': 0, 'finding': 1, 'query_result': 2}
    arts = sorted(snap['artifacts'], key=lambda a: (order.get(a[4], 3), a[0]))
    chunks, used = [], 0
    for i, ts, agent, name, kind, content in arts:
        block = f"### Artifact {i} [{kind}] by {agent}: {name}\n{content}\n"
        if used + len(block) > budget:
            block = block[:max(0, budget - used)]
        chunks.append(block)
        used += len(block)
        if used >= budget:
            break
    board = "\n".join(f"[{a}] {c}" for _, _, a, _, c in snap['board'])
    prompt = f"OBJECTIVE:\n{objective}\n\nBOARD:\n{board or '(empty)'}\n\nARTIFACTS:\n" + ("\n".join(chunks) or '(none)')
    return response_text(make_llm(llm_cfg).invoke([SystemMessage(content=REPORTER), HumanMessage(content=prompt)]))


def _execute(ws, objective, llm_cfg, n_agents, steps, dataset, do_synthesis, max_start_jitter, visibility='full'):
    """Run all agents for an already-created run, then (optionally) the observer report. Never raises."""
    try:
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futures = [ex.submit(run_agent, f'agent-{i + 1}', objective, ws, llm_cfg, steps, dataset,
                                 random.uniform(0, max_start_jitter) if i else 0.0, visibility)
                       for i in range(n_agents)]
            results = [f.result() for f in futures]  # run_agent never raises
        cancelled = ws.is_cancelled()
        report = None
        if do_synthesis and not cancelled:
            try:
                ws.trace('reporter', 'step_started', {'step': 'report'})
                report = synthesize(ws, objective, llm_cfg)
                ws.trace('reporter', 'run_finished', {'steps': 'report', 'done': True})
            except Exception as e:
                ws.trace('reporter', 'error', {'error': f"{type(e).__name__}: {e}"})
        failed = sum(1 for r in results if r.get('error'))
        ws.finish_run('cancelled' if cancelled else 'finished' if not failed else f'finished ({failed} agent(s) failed)', report)
        return {'run_id': ws.run_id, 'results': results, 'final_report': report}
    except BaseException as e:
        ws.trace('experiment', 'run_failed', {'error': f"{type(e).__name__}: {e}"})
        ws.finish_run('crashed')
        raise


def _setup(objective, llm_cfg, n_agents, steps, db_path, dataset, visibility='full'):
    if visibility not in VISIBILITY:
        raise ValueError(f"visibility must be one of {VISIBILITY}")
    llm_cfg = (llm_cfg or LLMConfig.from_env()).validate()
    base = Workspace(db_path)
    config = {'llm': llm_cfg.public(), 'n_agents': n_agents, 'steps': steps, 'visibility': visibility,
              'dataset': getattr(dataset, 'name', None),
              'tables': {n: len(d.df) for n, d in dataset.tables.items()} if dataset is not None else {}}
    return base, base.start_run(objective, config), llm_cfg


def run(objective, llm_cfg=None, n_agents=5, steps=6, db_path='workspace.sqlite', dataset=None,
        do_synthesis=True, max_start_jitter=3.0, visibility='full'):
    """Blocking run in its own run_id. Returns {'run_id', 'results', 'final_report'}.
    Agents start with a small random delay so they don't all act on an identical empty workspace."""
    base, ws, llm_cfg = _setup(objective, llm_cfg, n_agents, steps, db_path, dataset, visibility)
    try:
        return _execute(ws, objective, llm_cfg, n_agents, steps, dataset, do_synthesis, max_start_jitter, visibility)
    finally:
        base.close()


def launch(objective, llm_cfg=None, n_agents=5, steps=6, db_path='workspace.sqlite', dataset=None,
           do_synthesis=True, max_start_jitter=3.0, visibility='full'):
    """Non-blocking: start the run in a background thread and return (run_id, thread) immediately,
    so a UI can poll the workspace while agents work."""
    base, ws, llm_cfg = _setup(objective, llm_cfg, n_agents, steps, db_path, dataset, visibility)

    def target():
        try:
            _execute(ws, objective, llm_cfg, n_agents, steps, dataset, do_synthesis, max_start_jitter, visibility)
        except BaseException:
            pass  # already recorded as a run_failed trace
        finally:
            base.close()

    t = threading.Thread(target=target, name=f'run-{ws.run_id}', daemon=True)
    t.start()
    return ws.run_id, t
