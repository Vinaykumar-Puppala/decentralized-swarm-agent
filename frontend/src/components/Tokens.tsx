import { useMemo } from 'react'
import type { LlmCall, SwarmState } from '../types'
import { agentColor, fmtMs, fmtNum, fmtTokens } from '../util'
import { Who } from './bits'

const PURPOSE_LABEL: Record<string, string> = { decision: 'agent decisions', repair: 'JSON repair retries', 'report-map': 'report: condensing', 'report-merge': 'report: merging', report: 'report: final write' }

export default function Tokens({ shared, calls }: { shared: SwarmState | null; calls: LlmCall[] }) {
  const t = shared?.tokens
  const byPurpose = useMemo(() => {
    const m = new Map<string, { calls: number; tokens: number }>()
    for (const c of calls) { const e = m.get(c.purpose) ?? { calls: 0, tokens: 0 }; e.calls++; e.tokens += c.total_tokens; m.set(c.purpose, e) }
    return Array.from(m.entries()).sort((a, b) => b[1].tokens - a[1].tokens)
  }, [calls])
  const slowest = useMemo(() => calls.slice().sort((a, b) => b.latency_ms - a.latency_ms)[0], [calls])
  const biggest = useMemo(() => calls.slice().sort((a, b) => b.prompt_tokens - a.prompt_tokens)[0], [calls])
  if (!t || !t.total.calls) return <p className="empty">No model calls yet, so no tokens have been used.</p>

  const rows = Object.entries(t.agents).sort((a, b) => (a[0] === 'reporter' ? 1 : b[0] === 'reporter' ? -1 : a[0].localeCompare(b[0])))
  const max = Math.max(...rows.map(([, u]) => u.total), 1)
  const tot = t.total
  const tiles: [string, string, string?][] = [
    ['Total tokens', fmtTokens(tot.total)], ['Prompt (in)', fmtTokens(tot.prompt)], ['Completion (out)', fmtTokens(tot.completion)],
    ['Model calls', fmtNum(tot.calls)], ['Time in model', fmtMs(tot.seconds * 1000)], ['Failed calls', fmtNum(tot.errors), tot.errors ? 'bad' : undefined],
  ]
  return (
    <div className="stack">
      <div className="tiles">{tiles.map(([k, v, cls]) => <div key={k} className={`tile${cls ? ` ${cls}` : ''}`}><b>{v}</b><span>{k}</span></div>)}</div>
      {tot.estimated && <p className="muted small">Some counts are estimates (about 4 characters per token) because the provider did not report usage for those calls.</p>}

      <section className="paper wide">
        <h2>Tokens per agent</h2>
        <div className="legend small muted"><span><i className="sw in" /> prompt</span><span><i className="sw out" /> completion</span></div>
        <ul className="usage">
          {rows.map(([name, u]) => (
            <li key={name}>
              <div className="usage-name"><Who agent={name} /><span className="muted small">{fmtNum(u.calls)} {u.calls === 1 ? 'call' : 'calls'} · {((100 * u.total) / Math.max(tot.total, 1)).toFixed(0)}% of all</span></div>
              <div className="usage-bar" title={`${fmtNum(u.prompt)} in, ${fmtNum(u.completion)} out`} style={{ width: `${(100 * u.total) / max}%`, '--agent': agentColor(name) } as React.CSSProperties}>
                <div className="in" style={{ flex: u.prompt }} /><div className="out" style={{ flex: u.completion }} />
              </div>
              <div className="usage-num mono"><b>{fmtTokens(u.total)}</b><span className="muted small">{fmtTokens(u.prompt)} in · {fmtTokens(u.completion)} out{u.errors ? ` · ${u.errors} failed` : ''}</span></div>
            </li>
          ))}
        </ul>
      </section>

      <div className="two">
        <section className="paper wide">
          <h2>Where the tokens went</h2>
          <ul className="plain">
            {byPurpose.map(([p, v]) => <li key={p}><span>{PURPOSE_LABEL[p] ?? p}</span><span className="mono muted">{fmtNum(v.calls)} {v.calls === 1 ? 'call' : 'calls'}</span><b className="mono">{fmtTokens(v.tokens)}</b></li>)}
          </ul>
        </section>
        <section className="paper wide">
          <h2>Extremes</h2>
          <ul className="plain">
            {biggest && <li><span>Largest prompt</span><span className="muted small">{biggest.agent} · #{biggest.id}</span><b className="mono">{fmtTokens(biggest.prompt_tokens)}</b></li>}
            {slowest && <li><span>Slowest call</span><span className="muted small">{slowest.agent} · #{slowest.id}</span><b className="mono">{fmtMs(slowest.latency_ms)}</b></li>}
            <li><span>Average per call</span><span /><b className="mono">{fmtTokens(Math.round(tot.total / tot.calls))}</b></li>
          </ul>
        </section>
      </div>
    </div>
  )
}
