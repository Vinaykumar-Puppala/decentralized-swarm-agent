import random, threading
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import SystemMessage, HumanMessage
from .agent import VISIBILITY, run_agent
from .llm import LLMConfig, TrackedLLM, is_context_error, make_llm
from .workspace import Workspace

REPORTER = '''You are a post-hoc reporter. You did NOT take part in the experiment and must not add new analysis or numbers.
Summarise what the agents found into a clear report for a business reader: key insights (with the agents' computed numbers),
data-quality caveats, points where agents disagreed, and open questions. Attribute claims to agent IDs. Use Markdown.'''


MAPPER = '''You are condensing one part of a finished multi-agent run so a reporter can later write the final report.
Do NOT add analysis or numbers of your own. Keep every concrete finding, number, row id, caveat and disagreement, attribute each to its agent ID,
drop repetition and filler, and answer as compact Markdown bullet points.'''


def _pack(blocks, size):
    """Group text blocks so each group is at most `size` characters (a single oversized block is clipped to fit)."""
    groups, cur, used = [], [], 0
    for b in blocks:
        b = b if len(b) <= size else b[:size] + "\n[… clipped]"
        if cur and used + len(b) > size:
            groups.append(cur)
            cur, used = [], 0
        cur.append(b)
        used += len(b)
    return groups + ([cur] if cur else [])


def _report_once(llm, ws, objective, blocks, size):
    """Map-reduce over the run's material: condense each chunk, then (recursively) merge until it fits, then write the report."""
    rounds = 0
    while True:
        groups = _pack(blocks, size)
        if len(groups) == 1 or rounds >= 4:
            break
        ws.trace('reporter', 'report_progress', {'phase': 'condensing', 'round': rounds + 1, 'parts': len(groups)})
        blocks = [llm.call([SystemMessage(content=MAPPER), HumanMessage(content=f"OBJECTIVE:\n{objective}\n\nPART {i + 1} of {len(groups)}:\n" + "\n".join(g))],
                           'report-map' if rounds == 0 else 'report-merge')
                  for i, g in enumerate(groups)]
        rounds += 1
    ws.trace('reporter', 'report_progress', {'phase': 'writing', 'rounds': rounds})
    items = groups[0] if groups else []
    board = "\n".join(b for b in items if b.startswith('### Board')) if not rounds else "(merged into the notes below)"
    body = "\n".join(b for b in items if not b.startswith('### Board')) if not rounds else "\n".join(items)
    note = f"(The material was condensed in {rounds} pass(es) because it did not fit one model call.)\n" if rounds else ''
    return llm.call([SystemMessage(content=REPORTER),
                     HumanMessage(content=f"OBJECTIVE:\n{objective}\n\n{note}BOARD:\n{board or '(empty)'}\n\nARTIFACTS:\n{body or '(none)'}")], 'report')


def synthesize(ws, objective, llm_cfg, chunk_chars=None):
    """Observer-only summary of a finished run. It reads the workspace; agents never see its output.
    Material that does not fit one call is split, condensed piece by piece and merged. If the model still reports a context-limit
    error, the pieces are halved and the whole thing is retried."""
    llm = TrackedLLM(make_llm(llm_cfg), llm_cfg, ws, 'reporter')
    snap = ws.snapshot()
    order = {'final_answer': 0, 'finding': 1, 'query_result': 2}
    arts = sorted(snap['artifacts'], key=lambda a: (order.get(a[4], 3), a[0]))
    blocks = [f"### Board [{a}] {c}\n" for _, _, a, _, c in snap['board']]
    blocks += [f"### Artifact {i} [{kind}] by {agent}: {name}\n{content}\n" for i, ts, agent, name, kind, content in arts]
    size = int(chunk_chars or getattr(llm_cfg, 'context_chars', 0) or 24000)
    for attempt in range(4):
        try:
            return _report_once(llm, ws, objective, blocks, size)
        except Exception as e:
            if not is_context_error(e) or attempt == 3:
                raise
            size //= 2
            ws.trace('reporter', 'context_retry', {'error': str(e)[:300], 'chunk_chars': size})


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
