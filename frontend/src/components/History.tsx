import type { RunListItem } from '../types'
import { clip } from '../util'

export default function History({ runs, current, onPick }: { runs: RunListItem[]; current?: string; onPick: (id: string) => void }) {
  if (!runs.length) return null
  return (
    <section className="panel">
      <h2>Runs</h2>
      <ul className="history">
        {runs.slice(0, 30).map(r => (
          <li key={r.run_id}>
            <button className={r.run_id === current ? 'active' : ''} onClick={() => onPick(r.run_id)}>
              <span className={`dot ${r.active ? 'live' : r.status.startsWith('finished') ? 'ok' : 'off'}`} />
              <span className="h-main">{clip(r.objective, 56)}</span>
              <span className="h-sub muted">{new Date(r.started * 1000).toLocaleString([], { hour12: false })} · {r.status}</span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
