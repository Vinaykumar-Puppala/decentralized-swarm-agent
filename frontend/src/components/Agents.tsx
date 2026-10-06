import type { SwarmState } from '../types'
import { agentColor, fmtDuration, useNow } from '../util'

export default function Agents({ shared, steps, active }: { shared: SwarmState | null; steps: Record<string, string>; active: boolean }) {
  const now = useNow(active)
  if (!shared || !Object.keys(shared.agents).length) return <p className="empty">No agents have started yet.</p>
  const maxSteps = shared.config.steps ?? 1
  return (
    <div className="agents">
      {Object.entries(shared.agents).map(([name, a]) => {
        const waiting = a.state.startsWith('waiting') || a.state === 'writing report'
        const finished = ['finished', 'failed', 'stopped'].includes(a.state)
        const idle = !finished && a.lastTs ? now - a.lastTs : null
        return (
          <article key={name} className="agent" style={{ borderTopColor: agentColor(name) }}>
            <header>
              <b style={{ color: agentColor(name) }}>{name}</b>
              <span className={`state ${a.state.split(' ')[0]}`}>{waiting && <span className="pulse" />}{a.state}</span>
            </header>
            {name !== 'reporter' && (
              <div className="meter" title={`${a.steps} of ${maxSteps} steps`}><div style={{ width: `${Math.min(100, (100 * a.steps) / maxSteps)}%`, background: agentColor(name) }} /></div>
            )}
            <dl>
              {name !== 'reporter' && <><dt>steps</dt><dd>{a.steps} / {maxSteps}</dd><dt>rows read</dt><dd>{a.rowsRead.toLocaleString()}</dd></>}
              <dt>errors</dt><dd className={a.errors ? 'bad' : ''}>{a.errors}</dd>
              {steps[name] && <><dt>now</dt><dd className="mono">{steps[name]}</dd></>}
              {idle !== null && <><dt>last activity</dt><dd>{fmtDuration(idle)} ago</dd></>}
            </dl>
            {a.action && <p className="last"><b>{a.action}</b>{a.summary ? ` — ${a.summary}` : ''}</p>}
          </article>
        )
      })}
    </div>
  )
}
