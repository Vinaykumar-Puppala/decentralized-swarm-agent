import json, os, sys, time
from datetime import datetime
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, '.env'))
from swarm.data import DataCollection
from swarm.experiment import launch
from swarm.llm import DEFAULT_BASE_URLS, PROVIDERS, LLMConfig, ping
from swarm.workspace import Workspace

DB_PATH = os.getenv('WORKSPACE_DB', os.path.join(ROOT, 'workspace.sqlite'))
ACTIVE = ('running', 'cancelling')
ERROR_EVENTS = ('error', 'query_error', 'run_failed')

st.set_page_config(page_title='Decentralized Swarm V1', layout='wide')


@st.cache_resource
def workspace():
    return Workspace(DB_PATH)


@st.cache_resource
def threads():
    """run_id -> background thread, so we know which 'running' runs are really alive in this server process."""
    return {}


@st.cache_data(show_spinner=False)
def load_files(files: tuple):
    """files: tuple of (name, bytes). Returns a DataCollection (one table per file, or per Excel sheet)."""
    class _F:
        def __init__(s, n, b): s.name, s._b = n, b
        def getvalue(s): return s._b
    return DataCollection.from_sources([(_F(n, b), n) for n, b in files])


def hhmmss(ts):
    return datetime.fromtimestamp(ts).strftime('%H:%M:%S')


def clip(s, n):
    s = (s or '').replace('\n', ' ')
    return s if len(s) <= n else s[:n] + '…'


# ---------------- sidebar: model configuration
env = LLMConfig.from_env()
with st.sidebar:
    st.header('Model')
    provider = st.selectbox('Provider', PROVIDERS, index=PROVIDERS.index(env.provider) if env.provider in PROVIDERS else 0,
                            help='local = any OpenAI-compatible server (Ollama, LM Studio, vLLM). litellm = LiteLLM proxy.')
    model = st.text_input('Model name', env.model if provider == env.provider else '',
                          placeholder={'openai': 'gpt-4o-mini', 'anthropic': 'claude-sonnet-4-5', 'litellm': 'my-model-alias',
                                       'local': 'qwen2.5:3b'}[provider])
    base_url = st.text_input('Base URL (blank = default)', env.base_url if provider == env.provider else '',
                             placeholder=DEFAULT_BASE_URLS[provider])
    api_key = st.text_input('API key', env.api_key if provider == env.provider else '', type='password',
                            help='Not needed for most local servers. Never stored in the run log.')
    send_temp = st.checkbox('Send temperature', value=env.temperature is not None,
                            help='Untick for reasoning models that reject a temperature parameter.')
    temperature = st.slider('Temperature', 0.0, 1.5, env.temperature if env.temperature is not None else 0.7, 0.1,
                            disabled=not send_temp)
    cfg = LLMConfig(provider=provider, model=model.strip(), api_key=api_key.strip(), base_url=base_url.strip(),
                    temperature=temperature if send_temp else None, max_tokens=env.max_tokens, timeout=env.timeout)
    if st.button('Test connection'):
        try:
            with st.spinner('Calling model...'):
                t = time.time()
                reply = ping(cfg)
            st.success(f'Connected in {time.time() - t:.1f}s. Reply: {reply[:80]!r}')
        except Exception as e:
            st.error(f'{type(e).__name__}: {e}')

ws_all = workspace()
alive = threads()
# runs left 'running' by a previous server process can never finish: mark them so they don't block the UI
for r in ws_all.runs():
    if r[3] in ACTIVE and not (r[0] in alive and alive[r[0]].is_alive()):
        ws_all.for_run(r[0]).finish_run('interrupted')
runs = ws_all.runs()
active_id = next((r[0] for r in runs if r[3] in ACTIVE), None)

# ---------------- main: experiment setup
st.title('Decentralized Multi-Agent Shared Workspace — V1')
st.caption('Identical agents • no orchestrator • no direct agent chat • shared environment only')
objective = st.text_area('Common objective', 'Explore this data and give deeper insights.', height=100)
c1, c2, c3 = st.columns(3)
n_agents = c1.slider('Agents', 1, 8, 5)
steps = c2.slider('Maximum steps per agent', 1, 40, 10,
                  help='Each data_rows call reads up to 100 rows, so more steps = more of the data actually read.')
do_report = c3.checkbox('Write post-hoc report after the run', value=True,
                        help='An observer summarises the workspace once all agents finish. Agents never see it.')

dataset = None
files = st.file_uploader('Optional data: CSV / Excel / Parquet (select several; each file, and each Excel sheet, becomes its own table, not joined)',
                         type=['csv', 'xlsx', 'xlsm', 'xls', 'parquet'], accept_multiple_files=True)
if files:
    try:
        dataset = load_files(tuple((f.name, f.getvalue()) for f in files))
        tabs = st.tabs([f"{n} ({d.df.shape[0]}×{d.df.shape[1]})" for n, d in dataset.tables.items()])
        for tab, (n, d) in zip(tabs, dataset.tables.items()):
            with tab:
                st.dataframe(d.df.head(100))
        with st.expander('Profile given to every agent'):
            st.text(dataset.profile)
    except Exception as e:
        st.error(f'Could not read files: {type(e).__name__}: {e}')

b1, b2, _ = st.columns([2, 1, 5])
if b1.button(f'Run {n_agents}-agent experiment', type='primary', disabled=active_id is not None,
             help='Another run is still active.' if active_id else None):
    try:
        cfg.validate()
        run_id, thread = launch(objective, cfg, n_agents, steps, DB_PATH, dataset, do_synthesis=do_report)
        alive[run_id] = thread
        st.session_state['run_id'] = run_id
        st.rerun()
    except ValueError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f'Could not start: {type(e).__name__}: {e}')
if b2.button('Stop run', disabled=active_id is None):
    ws_all.for_run(active_id).request_cancel()
    st.toast('Stopping: agents finish their current model call, then halt.')

if not runs:
    st.info('No runs yet.')
    st.stop()
st.divider()
labels = {r[0]: f"{r[0]} · {r[3]} · {r[2][:60]}" for r in runs}
ids = list(labels)
current = st.session_state.get('run_id') or ids[0]
if active_id and 'run_id' not in st.session_state:
    current = active_id
run_id = st.selectbox('Run', ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
st.session_state['run_id'] = run_id


# ---------------- live view
def feed_lines(s):
    """Merge traces, board posts and artifacts into one chronological activity feed (ts, markdown)."""
    ev = []
    for _id, ts, agent, event, payload in s['traces']:
        p = json.loads(payload or '{}')
        if event == 'step_started':
            ev.append((ts, f"⏳ **{agent}** step {p.get('step')}: calling the model…"))
        elif event == 'decision':
            conf = f" · confidence {p['confidence']:.2f}" if p.get('confidence') is not None else ''
            extra = ''
            if p.get('data_query'):
                extra += f"\n  - 🔎 query: `{clip(json.dumps(p['data_query']), 300)}`"
            if p.get('data_rows'):
                extra += f"\n  - 📋 reads rows: `{clip(json.dumps(p['data_rows']), 300)}`"
            if p.get('artifact_to_read') is not None:
                extra += f"\n  - 📖 reads artifact #{p['artifact_to_read']}"
            ev.append((ts, f"🧠 **{agent}** step {p.get('step')} · *{p.get('action') or '—'}*{conf} — {p.get('audit_summary') or ''}{extra}"))
        elif event in ERROR_EVENTS:
            ev.append((ts, f"🛑 **{agent}** {event}: `{clip(p.get('error', ''), 300)}`"))
        elif event == 'rows_viewed':
            ev.append((ts, f"👁 **{agent}** got {p.get('count')} rows of `{p.get('table')}` (row ids {p.get('first')}–{p.get('last')})"))
        elif event == 'cancelled':
            ev.append((ts, f"⏹ **{agent}** stopped on request"))
        elif event == 'run_started':
            ev.append((ts, f"▶️ **{agent}** started (max {p.get('max_steps')} steps)"))
        elif event == 'run_finished':
            ev.append((ts, f"🏁 **{agent}** finished" + ("" if agent == 'reporter' else f" after {p.get('steps')} steps (done={p.get('done')})")))
    for _id, ts, agent, kind, content in s['board']:
        ev.append((ts, f"📣 **{agent}** posted to the board: {content}"))
    for _id, ts, agent, name, kind, content in s['artifacts']:
        ev.append((ts, f"📄 **{agent}** published artifact #{_id} [{kind}] *{clip(name, 80)}*: {clip(content, 260)}"))
    return [f"`{hhmmss(ts)}` {txt}" for ts, txt in sorted(ev, key=lambda e: e[0], reverse=True)]


def agent_table(s, info, cov=None):
    now, rows, agents = time.time(), [], {}
    for _id, ts, agent, event, payload in s['traces']:
        a = agents.setdefault(agent, {'steps': 0, 'last_ts': ts, 'last': event, 'action': '', 'summary': '', 'state': 'working', 'errors': 0})
        a['last_ts'], p = ts, json.loads(payload or '{}')
        if event == 'step_started':
            a['state'] = f"waiting on model (step {p.get('step')})"
        elif event == 'decision':
            a.update(steps=a['steps'] + 1, action=p.get('action') or '', summary=p.get('audit_summary') or '', state='working')
        elif event in ERROR_EVENTS:
            a['errors'] += 1
        elif event == 'run_finished':
            a['state'] = 'finished'
        elif event == 'run_failed':
            a['state'] = 'failed'
        elif event == 'cancelled':
            a['state'] = 'stopped'
    for agent, a in agents.items():
        idle = '' if a['state'] in ('finished', 'failed', 'stopped') else f"{now - a['last_ts']:.0f}s ago"
        read = sum(per.get(agent, 0) for per in (cov or {}).values())
        rows.append({'agent': agent, 'state': a['state'], 'steps done': a['steps'], 'rows read': read, 'errors': a['errors'],
                     'last activity': idle, 'last action': a['action'], 'last summary': a['summary']})
    return pd.DataFrame(rows)


def live_view():
    ws = ws_all.for_run(run_id)
    info = ws.run_info()
    status = info[4]
    s = ws.snapshot()
    traces = pd.DataFrame(s['traces'], columns=['id', 'timestamp', 'agent', 'event', 'payload'])
    arts = pd.DataFrame(s['artifacts'], columns=['id', 'timestamp', 'agent', 'name', 'kind', 'content'])
    acc = pd.DataFrame(s['accesses'], columns=['id', 'timestamp', 'reader', 'artifact_id', 'author', 'cross_agent', 'action'])

    if status in ACTIVE:
        st.info(f"⏳ Run **{status}** — refreshing every 2 s. Started {hhmmss(info[1])}, elapsed {time.time() - info[1]:.0f}s.")
    else:
        if status.startswith('finished'):
            st.success(f"Run **{status}**.")
        else:
            st.warning(f"Run **{status}**.")

    cov = ws.coverage_counts()
    table_sizes = json.loads(info[3] or '{}').get('tables', {})
    total_rows = sum(table_sizes.values())
    seen_rows = sum(min(cov.get(t, {}).get('all', 0), n) for t, n in table_sizes.items())
    if total_rows:
        st.progress(seen_rows / total_rows, text=f"Rows read by the swarm: {seen_rows:,} of {total_rows:,} ({100 * seen_rows / total_rows:.0f}%)")
    m = st.columns(6)
    m[0].metric('Board posts', len(s['board']))
    m[1].metric('Findings / finals', f"{(arts.kind == 'finding').sum()} / {(arts.kind == 'final_answer').sum()}")
    m[2].metric('Data queries', int((arts.kind == 'query_result').sum()))
    m[3].metric('Cross-agent reads', int(acc.cross_agent.fillna(0).sum()) if len(acc) else 0)
    m[4].metric('Errors', int(traces.event.isin(ERROR_EVENTS).sum()) if len(traces) else 0)
    m[5].metric('Trace events', len(traces))

    t_feed, t_agents, t_board, t_art, t_final, t_raw = st.tabs(
        ['Live feed', 'Agents', 'Message board', 'Artifacts', 'Report & final answers', 'Raw tables'])
    with t_feed:
        st.caption('Newest first. "🧠" lines are each agent\'s audit summary: observation, action and high-level reason, not hidden chain-of-thought.')
        lines = feed_lines(s)
        st.markdown('\n\n'.join(lines[:150]) if lines else '_Waiting for the first agent event…_')
    with t_agents:
        df = agent_table(s, info, cov)
        if len(df):
            st.dataframe(df, hide_index=True)
        else:
            st.write('No agents yet.')
    with t_board:
        board = s['board'][::-1]
        if not board:
            st.write('No board posts yet.')
        for _id, ts, agent, kind, content in board:
            st.markdown(f"`{hhmmss(ts)}` **{agent}**: {content}")
    with t_art:
        if not len(arts):
            st.write('No artifacts yet.')
        for _, r in arts.iloc[::-1].iterrows():
            with st.expander(f"#{r['id']} · {r.agent} · {r.kind} · {clip(r['name'], 80)}"):
                st.text(r.content)
    with t_final:
        if info[5]:
            st.subheader('Post-hoc report (observer, not an agent)')
            st.markdown(info[5])
        finals = arts[arts.kind == 'final_answer']
        for _, r in finals.iterrows():
            with st.expander(f"{r.agent}: {r['name']}", expanded=not info[5]):
                st.markdown(r.content)
        if not info[5] and not len(finals):
            st.write('Nothing yet.')
    with t_raw:
        errs = traces[traces.event.isin(ERROR_EVENTS)] if len(traces) else traces
        if len(errs):
            st.subheader('Errors')
            st.dataframe(errs)
        with st.expander('Run configuration'):
            st.json(json.loads(info[3] or '{}'))
        st.subheader('Artifact access events')
        st.dataframe(acc)
        st.subheader('Agent audit traces')
        st.dataframe(traces)

    # when the run ends, do one full-page rerun so the Run/Stop buttons and run list update
    if st.session_state.get('_last_status') in ACTIVE and status not in ACTIVE:
        st.session_state['_last_status'] = status
        st.rerun()
    st.session_state['_last_status'] = status


status_now = ws_all.for_run(run_id).status()
st.fragment(live_view, run_every=2 if status_now in ACTIVE else None)()
