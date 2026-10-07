import { useMemo, useRef, useState } from 'react'
import { llmCall } from '../api'
import type { LlmCall, LlmCallDetail } from '../types'
import { agentColor, fmtMs, fmtNum, fmtTime } from '../util'
import { Tag, Who, type Tone } from './bits'

const PURPOSE: Record<string, [string, Tone]> = {
  decision: ['decision', 'info'], repair: ['repair', 'warn'], 'report-map': ['report · condense', 'accent'],
  'report-merge': ['report · merge', 'accent'], report: ['report', 'ok'],
}
const ROLE_TONE: Record<string, Tone> = { system: 'accent', user: 'info', assistant: 'ok' }

function copy(text: string) { void navigator.clipboard?.writeText(text).catch(() => undefined) }

function Detail({ runId, call }: { runId: string; call: LlmCall }) {
  const [d, setD] = useState<LlmCallDetail | null>(null)
  const [err, setErr] = useState('')
  const started = useRef(false)
  if (!started.current) {
    started.current = true
    llmCall(runId, call.id).then(setD).catch(e => setErr(String(e)))
  }
  if (err) return <p className="note bad">{err}</p>
  if (!d) return <p className="muted small">Loading…</p>
  const input = d.input.map(m => `[${m.role}]\n${m.content}`).join('\n\n')
  return (
    <div className="call-detail">
      {d.error && <div className="banner bad small"><b>Error</b> <span className="mono">{d.error}</span></div>}
      <div className="call-section">
        <div className="call-section-head"><h2>Input · {d.input.length} messages · {fmtNum(call.in_chars)} characters</h2><button className="link" onClick={() => copy(input)}>copy</button></div>
        {d.input.map((m, i) => (
          <div key={i} className="msg"><Tag tone={ROLE_TONE[m.role] ?? 'neutral'}>{m.role}</Tag><pre>{m.content}</pre></div>
        ))}
      </div>
      <div className="call-section">
        <div className="call-section-head"><h2>Output · {fmtNum(call.out_chars)} characters</h2><button className="link" onClick={() => copy(d.output)}>copy</button></div>
        {d.output ? <pre>{d.output}</pre> : <p className="muted small">No reply (the call failed).</p>}
      </div>
    </div>
  )
}

export default function LlmLog({ calls, runId }: { calls: LlmCall[]; runId: string | undefined }) {
  const [agent, setAgent] = useState('all')
  const [purpose, setPurpose] = useState('all')
  const [errorsOnly, setErrorsOnly] = useState(false)
  const [open, setOpen] = useState<number | null>(null)
  const agents = useMemo(() => Array.from(new Set(calls.map(c => c.agent))).sort((a, b) => (a === 'reporter' ? 1 : b === 'reporter' ? -1 : a.localeCompare(b))), [calls])
  const purposes = useMemo(() => Array.from(new Set(calls.map(c => c.purpose))), [calls])
  const rows = useMemo(
    () => calls.filter(c => (agent === 'all' || c.agent === agent) && (purpose === 'all' || c.purpose === purpose) && (!errorsOnly || c.status !== 'ok')).slice(-300).reverse(),
    [calls, agent, purpose, errorsOnly])

  if (!calls.length) return <p className="empty">No model calls yet.</p>
  return (
    <div>
      <div className="toolbar">
        <div className="chips">
          <button className={agent === 'all' ? 'chip on' : 'chip'} onClick={() => setAgent('all')}>All</button>
          {agents.map(a => (
            <button key={a} className={agent === a ? 'chip on' : 'chip'} onClick={() => setAgent(a)}><span className="swatch" style={{ background: agentColor(a) }} />{a}</button>
          ))}
        </div>
        <div className="toolbar-right">
          <select aria-label="Filter by purpose" value={purpose} onChange={e => setPurpose(e.target.value)}>
            <option value="all">All purposes</option>
            {purposes.map(p => <option key={p} value={p}>{PURPOSE[p]?.[0] ?? p}</option>)}
          </select>
          <label className="row small"><input type="checkbox" checked={errorsOnly} onChange={e => setErrorsOnly(e.target.checked)} /> errors only</label>
        </div>
      </div>
      <p className="muted small">Every call to the model, newest first: the exact messages sent and the reply received. Click a row to read them. Token counts marked “est.” are estimated (about 4 characters per token) because the provider did not report usage.</p>
      <div className="calls">
        {rows.map(c => (
          <article key={c.id} className={`call${c.status !== 'ok' ? ' is-bad' : ''}${open === c.id ? ' open' : ''}`} style={{ '--agent': agentColor(c.agent) } as React.CSSProperties}>
            <button className="call-row" aria-expanded={open === c.id} onClick={() => setOpen(open === c.id ? null : c.id)}>
              <span className="mono muted call-id">#{c.id}</span>
              <Who agent={c.agent} />
              <Tag tone={PURPOSE[c.purpose]?.[1] ?? 'neutral'}>{PURPOSE[c.purpose]?.[0] ?? c.purpose}</Tag>
              {c.step != null && <span className="muted small">step {c.step}</span>}
              {c.status !== 'ok' && <Tag tone="bad">failed</Tag>}
              <span className="call-tokens mono">{fmtNum(c.prompt_tokens)} <span className="muted">in</span> → {fmtNum(c.completion_tokens)} <span className="muted">out</span>{c.estimated && <span className="muted"> est.</span>}</span>
              <span className="muted small mono call-lat">{fmtMs(c.latency_ms)}</span>
              <time className="muted mono small">{fmtTime(c.ts)}</time>
            </button>
            {open === c.id && runId && <Detail key={c.id} runId={runId} call={c} />}
          </article>
        ))}
      </div>
    </div>
  )
}
