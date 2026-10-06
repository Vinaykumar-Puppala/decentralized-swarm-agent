import random
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage
from .agent import run_agent
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


def run(objective, llm_cfg=None, n_agents=5, steps=6, db_path='workspace.sqlite', dataset=None,
        do_synthesis=True, max_start_jitter=3.0):
    """Run one experiment in its own run_id. Returns {'run_id', 'results', 'final_report'}.
    Agents start with a small random delay so they don't all act on an identical empty workspace."""
    llm_cfg = (llm_cfg or LLMConfig.from_env()).validate()
    base = Workspace(db_path)
    config = {'llm': llm_cfg.public(), 'n_agents': n_agents, 'steps': steps,
              'dataset': getattr(dataset, 'name', None)}
    ws = base.start_run(objective, config)
    try:
        with ThreadPoolExecutor(max_workers=n_agents) as ex:
            futures = [ex.submit(run_agent, f'agent-{i + 1}', objective, ws, llm_cfg, steps, dataset,
                                 random.uniform(0, max_start_jitter) if i else 0.0)
                       for i in range(n_agents)]
            results = [f.result() for f in futures]  # run_agent never raises
        report = None
        if do_synthesis:
            try:
                report = synthesize(ws, objective, llm_cfg)
            except Exception as e:
                ws.trace('reporter', 'error', {'error': f"{type(e).__name__}: {e}"})
        failed = sum(1 for r in results if r.get('error'))
        ws.finish_run('finished' if not failed else f'finished ({failed} agent(s) failed)', report)
        return {'run_id': ws.run_id, 'results': results, 'final_report': report}
    except BaseException:
        ws.finish_run('crashed')
        raise
    finally:
        base.close()
