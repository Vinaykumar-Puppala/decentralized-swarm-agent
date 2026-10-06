import json, os, sys, time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
load_dotenv(os.path.join(ROOT, '.env'))
from swarm.data import Dataset
from swarm.experiment import run
from swarm.llm import DEFAULT_BASE_URLS, PROVIDERS, LLMConfig, ping
from swarm.workspace import Workspace

DB_PATH = os.getenv('WORKSPACE_DB', os.path.join(ROOT, 'workspace.sqlite'))

st.set_page_config(page_title='Decentralized Swarm V1', layout='wide')


@st.cache_resource
def workspace():
    return Workspace(DB_PATH)


@st.cache_data(show_spinner=False)
def load_dataset(data: bytes, name: str):
    class _F:  # minimal file-like for load_table
        def __init__(s): s.name = name
        def getvalue(s): return data
    return Dataset.from_source(_F(), name)


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

# ---------------- main: experiment setup
st.title('Decentralized Multi-Agent Shared Workspace — V1')
st.caption('Identical agents • no orchestrator • no direct agent chat • shared environment only')
objective = st.text_area('Common objective', 'Explore this data and give deeper insights.', height=100)
c1, c2, c3 = st.columns(3)
n_agents = c1.slider('Agents', 1, 8, 5)
steps = c2.slider('Maximum steps per agent', 1, 12, 6)
do_report = c3.checkbox('Write post-hoc report after the run', value=True,
                        help='An observer summarises the workspace once all agents finish. Agents never see it.')

dataset = None
file = st.file_uploader('Optional CSV / Excel / Parquet context', type=['csv', 'xlsx', 'xls', 'parquet'])
if file:
    try:
        dataset = load_dataset(file.getvalue(), file.name)
        st.write(f'**{file.name}**: {dataset.df.shape[0]} rows × {dataset.df.shape[1]} columns after cleaning')
        st.dataframe(dataset.df.head(100))
        with st.expander('Profile given to every agent'):
            st.text(dataset.profile)
    except Exception as e:
        st.error(f'Could not read file: {type(e).__name__}: {e}')

if st.button(f'Run {n_agents}-agent experiment', type='primary'):
    try:
        cfg.validate()
    except ValueError as e:
        st.error(str(e))
        st.stop()
    t = time.time()
    with st.spinner(f'Running {n_agents} decentralized agents (up to {n_agents * steps} model calls)...'):
        try:
            out = run(objective, cfg, n_agents, steps, DB_PATH, dataset, do_synthesis=do_report)
            st.session_state['run_id'] = out['run_id']
            st.success(f"Run {out['run_id']} finished in {time.time() - t:.0f}s.")
        except Exception as e:
            st.error(f'Experiment failed: {type(e).__name__}: {e}')

# ---------------- results
ws_all = workspace()
runs = ws_all.runs()
if not runs:
    st.info('No runs yet.')
    st.stop()
st.divider()
labels = {r[0]: f"{r[0]} · {r[3]} · {r[2][:60]}" for r in runs}
ids = list(labels)
current = st.session_state.get('run_id', ids[0])
run_id = st.selectbox('Run', ids, index=ids.index(current) if current in ids else 0, format_func=labels.get)
ws = ws_all.for_run(run_id)
info = ws.run_info()
s = ws.snapshot()

traces = pd.DataFrame(s['traces'], columns=['id', 'timestamp', 'agent', 'event', 'payload'])
arts = pd.DataFrame(s['artifacts'], columns=['id', 'timestamp', 'agent', 'name', 'kind', 'content'])
acc = pd.DataFrame(s['accesses'], columns=['id', 'timestamp', 'reader', 'artifact_id', 'author', 'cross_agent', 'action'])
m = st.columns(6)
m[0].metric('Board posts', len(s['board']))
m[1].metric('Findings / finals', f"{(arts.kind == 'finding').sum()} / {(arts.kind == 'final_answer').sum()}")
m[2].metric('Data queries', int((arts.kind == 'query_result').sum()))
m[3].metric('Cross-agent reads', int(acc.cross_agent.fillna(0).sum()) if len(acc) else 0)
m[4].metric('Errors', int(traces.event.isin(['error', 'query_error', 'run_failed']).sum()))
m[5].metric('Trace events', len(traces))

with st.expander('Run configuration'):
    st.json(json.loads(info[3] or '{}'))
if info and info[5]:
    st.subheader('Post-hoc report (observer, not an agent)')
    st.markdown(info[5])

finals = arts[arts.kind == 'final_answer']
if len(finals):
    st.subheader('Agents\' final answers')
    for _, r in finals.iterrows():
        with st.expander(f"{r.agent}: {r['name']}"):
            st.markdown(r.content)

errs = traces[traces.event.isin(['error', 'query_error', 'run_failed'])]
if len(errs):
    st.subheader('Errors')
    st.dataframe(errs)

st.subheader('Shared message board')
st.dataframe(pd.DataFrame(s['board'], columns=['id', 'timestamp', 'agent', 'kind', 'content']))
st.subheader('Artifacts')
st.dataframe(arts)
st.subheader('Artifact access events')
st.dataframe(acc)
st.subheader('Agent audit traces')
st.dataframe(traces)
