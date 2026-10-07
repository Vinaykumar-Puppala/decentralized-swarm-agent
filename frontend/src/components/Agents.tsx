import type { SwarmState } from '../types'
import { agentColor, fmtDuration, useNow } from '../util'

const initial = (name: string) => (name === 'reporter' ? 'R' : name.replace(/\D/g, '') || name[0].toUpperCase())

export default function Agents({ shared, steps, active }: { shared: SwarmState | null; steps: Record<string, string>; active: boolean }) {
  const now = useNow(active)
  if (!shared || !Object.keys(shared.agents).length) return <p className="empty">No agents have started yet.</p>
  const maxSteps = shared.config.steps ?? 1
  return (
    <div className="agents">
      {Object.entries(shared.agents).map(([name, a]) => {
        const color = agentColor(name)
        const working = a.state.startsWith('waiting') || a.state === 'writing report' || a.state === 'working'
        const finished = ['finished', 'failed', 'stopped'].includes(a.state)
        const idle = !finished && a.lastTs ? now - a.lastTs : null
        const isReporter = name === 'reporter'
        return (
          <article key={name} className={`agent${working ? ' working' : ''}${a.state === 'failed' ? ' failed' : ''}`} style={{ '--agent': color } as React.CSSProperties}>
            <header>
              <span className="avatar" aria-hidden>{initial(name)}</span>
              <div className="agent-title">
                <b>{name}</b>
                <span className="muted small">{isReporter ? 'post-hoc observer' : steps[name] ? `now: ${steps[name]}` : finished ? 'idle' : 'thinking…'}</span>
              </div>
              <span className={`state ${a.state.split(' ')[0]}`}>{working && <span className="pulse" />}{a.state}</span>
            </header>
            {!isReporter && (
              <div className="progress-row">
                <div className="meter" role="progressbar" aria-valuemin={0} aria-valuemax={maxSteps} aria-valuenow={a.steps}><div style={{ width: `${Math.min(100, (100 * a.steps) / maxSteps)}%` }} /></div>
                <span className="small muted mono">{a.steps}/{maxSteps}</span>
              </div>
            )}
            <div className="stats">
              {!isReporter && <div><b>{a.rowsRead.toLocaleString()}</b><span>rows read</span></div>}
              <div className={a.errors ? 'bad' : ''}><b>{a.errors}</b><span>errors</span></div>
              {idle !== null && <div><b>{fmtDuration(idle)}</b><span>since last</span></div>}
            </div>
            {a.action && <p className="last"><b>{a.action}</b>{a.summary ? ` — ${a.summary}` : ''}</p>}
          </article>
        )
      })}
    </div>
  )
}
