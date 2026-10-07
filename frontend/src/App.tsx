import { useCallback, useEffect, useMemo, useState } from 'react'
import { getConfig, listRuns, stopRun } from './api'
import Agents from './components/Agents'
import DataBrowser from './components/DataBrowser'
import DataPanel from './components/DataPanel'
import LlmLog from './components/LlmLog'
import Tokens from './components/Tokens'
import Feed from './components/Feed'
import History from './components/History'
import ModelPanel from './components/ModelPanel'
import Report from './components/Report'
import RunHeader from './components/RunHeader'
import RunPanel from './components/RunPanel'
import { ArtifactsView, BoardView } from './components/Workspace'
import { useSwarmRun } from './swarm/useSwarmRun'
import type { AppConfig, DatasetInfo, LLMForm, RunForm, RunListItem } from './types'
import ThemeToggle from './components/ThemeToggle'
import { useLocalStorage, useTheme } from './util'

type Tab = 'feed' | 'agents' | 'board' | 'artifacts' | 'llm' | 'tokens' | 'data' | 'report'
const TABS: [Tab, string][] = [['feed', 'Live feed'], ['agents', 'Agents'], ['board', 'Board'], ['artifacts', 'Artifacts'], ['llm', 'LLM calls'], ['tokens', 'Tokens'], ['data', 'Data'], ['report', 'Report']]

type SavedLLM = Omit<LLMForm, 'api_key'>

export default function App() {
  const [cfg, setCfg] = useState<AppConfig | null>(null)
  const [saved, setSaved] = useLocalStorage<SavedLLM>('swarm.llm', { provider: 'local', model: '', base_url: '', temperature: 0.7 })
  const [apiKey, setApiKey] = useState('')                 // the key is never written to localStorage
  const [run, setRun] = useLocalStorage<RunForm>('swarm.run', { objective: 'Explore this data and give deeper insights.', nAgents: 5, steps: 10, report: true })
  const [dataset, setDataset] = useState<DatasetInfo | null>(null)
  const [runs, setRuns] = useState<RunListItem[]>([])
  const [tab, setTab] = useState<Tab>('feed')
  const [stopping, setStopping] = useState(false)
  const [notice, setNotice] = useState('')
  const [theme, setTheme] = useTheme()

  const refreshRuns = useCallback(() => { listRuns().then(setRuns).catch(() => undefined) }, [])
  const swarm = useSwarmRun(refreshRuns)
  const { data } = swarm

  // first load: server defaults (without overriding what the user already saved) and the run history
  useEffect(() => {
    getConfig().then(c => {
      setCfg(c)
      setSaved(s => (s.model || s.base_url ? s : { ...s, provider: c.defaults.provider, model: c.defaults.model, base_url: c.defaults.base_url, temperature: c.defaults.temperature }))
    }).catch(e => setNotice(`Cannot reach the API: ${e}`))
    listRuns().then(list => {
      setRuns(list)
      const active = list.find(r => r.active)
      if (active) void swarm.attach(active.run_id)       // page reload while a run is going: re-attach to its live stream
    }).catch(() => undefined)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => { if (!swarm.busy) setStopping(false) }, [swarm.busy])
  useEffect(() => { if (swarm.busy) refreshRuns() }, [swarm.busy, refreshRuns])

  const llm: LLMForm = useMemo(() => ({ ...saved, api_key: apiKey }), [saved, apiKey])
  const modelKnown = !!llm.model || (!!cfg?.defaults.model && cfg.defaults.provider === llm.provider)
  const hint = !modelKnown ? 'Enter a model name to start.' : !run.objective.trim() ? 'Write an objective to start.' : undefined

  const agents = useMemo(() => Object.keys(data.shared?.agents ?? {}), [data.shared])
  const reportReady = !!(data.result?.report || data.feed.some(f => f.kind === 'message' && f.agent === 'reporter'))
  const currentRun = data.shared?.swarmRunId
  const boardCount = useMemo(() => data.feed.filter(f => f.kind === 'board').length, [data.feed])

  async function stop() {
    if (!currentRun) return
    setStopping(true)
    try { await stopRun(currentRun) } catch (e) { setNotice(String(e)); setStopping(false) }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <svg width="26" height="26" viewBox="0 0 32 32" aria-hidden><circle cx="9" cy="10" r="4" fill="#6d5efc" /><circle cx="23" cy="10" r="4" fill="#22b8a6" /><circle cx="16" cy="23" r="4" fill="#f5a524" /><path d="M9 10 23 10 16 23Z" fill="none" stroke="currentColor" strokeOpacity=".35" /></svg>
          <div className="brand-text"><b>Decentralized Swarm</b><span className="muted small">no orchestrator · shared workspace</span></div>
          <ThemeToggle theme={theme} onChange={setTheme} />
        </div>
        <ModelPanel cfg={cfg} value={llm} onChange={v => { const { api_key, ...rest } = v; setApiKey(api_key); setSaved(rest) }} />
        <DataPanel dataset={dataset} onDataset={setDataset} locked={swarm.busy} />
        <RunPanel value={run} onChange={setRun} busy={swarm.busy} canStart={modelKnown && !!run.objective.trim()} hint={hint} stopping={stopping}
                  onStart={() => { setNotice(''); setTab('feed'); void swarm.start(llm, run, dataset?.dataset_id ?? null) }} onStop={stop} />
        <History runs={runs} current={currentRun} onPick={id => { setTab('feed'); void swarm.attach(id) }} />
      </aside>

      <main className="main">
        {notice && <div className="banner bad" role="alert">{notice}<button className="link" onClick={() => setNotice('')}>dismiss</button></div>}
        {data.error && <div className="banner bad" role="alert"><b>Run failed.</b> {data.error}</div>}
        {data.phase === 'idle' && !data.shared ? (
          <div className="hero">
            <h1>Let a swarm of identical agents dig into your question</h1>
            <p>There is no planner and no assigned roles. Each agent decides for itself what to look at next, reading the data, querying it, and using a shared board and artifacts only when that helps.</p>
            <ol>
              <li>Pick a model (hosted API, LiteLLM proxy, or a local server) and press <b>Test connection</b>.</li>
              <li>Optionally upload several CSV / Excel / Parquet files: each file or sheet is its own table, and agents can read every row.</li>
              <li>Write the objective and run. Everything the agents do streams in live over the AG-UI protocol.</li>
            </ol>
          </div>
        ) : (
          <>
            <RunHeader data={data} />
            <nav className="tabs" role="tablist">
              {TABS.map(([id, label]) => (
                <button key={id} role="tab" aria-selected={tab === id} className={tab === id ? 'tab on' : 'tab'} onClick={() => setTab(id)}>
                  {label}{id === 'report' && reportReady && <span className="dot ok" />}
                  {id === 'board' && boardCount ? <span className="count">{boardCount}</span> : null}
                  {id === 'llm' && data.calls.length ? <span className="count">{data.calls.length}</span> : null}
                  {id === 'agents' && agents.length ? <span className="count">{agents.length}</span> : null}
                </button>
              ))}
            </nav>
            <div className="tabpanel" role="tabpanel" key={tab}>
              {tab === 'feed' && <Feed feed={data.feed} agents={agents} live={swarm.busy} />}
              {tab === 'agents' && <Agents shared={data.shared} steps={data.steps} active={swarm.busy} />}
              {tab === 'board' && <BoardView feed={data.feed} />}
              {tab === 'artifacts' && <ArtifactsView feed={data.feed} />}
              {tab === 'llm' && <LlmLog calls={data.calls} runId={currentRun} />}
              {tab === 'tokens' && <Tokens shared={data.shared} calls={data.calls} />}
              {tab === 'data' && <DataBrowser dataset={dataset} shared={data.shared} />}
              {tab === 'report' && <Report data={data} />}
            </div>
          </>
        )}
      </main>
    </div>
  )
}
