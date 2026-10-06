import type { RunData } from '../types'
import { fmtDuration, useNow } from '../util'

const LABEL: Record<RunData['phase'], string> = {
  idle: 'No run', connecting: 'Connecting…', running: 'Running', finished: 'Finished', cancelled: 'Stopped', error: 'Error',
}

export default function RunHeader({ data }: { data: RunData }) {
  const live = data.phase === 'running' || data.phase === 'connecting'
  const now = useNow(live)
  const s = data.shared
  if (!s) return null
  const last = Math.max(s.startedAt, ...Object.values(s.agents).map(a => a.lastTs || 0))
  const elapsed = (live ? now : last) - s.startedAt
  const total = Object.values(s.coverage).reduce((n, c) => n + c.total, 0)
  const read = Object.values(s.coverage).reduce((n, c) => n + c.read, 0)
  const c = s.counts
  const tiles: [string, string | number][] = [
    ['Board posts', c.board], ['Findings', c.findings], ['Final answers', c.finals], ['Data queries', c.queries],
    ['Row reads', c.row_reads], ['Cross-agent reads', c.cross_reads], ['Errors', c.errors],
  ]
  return (
    <header className="runhead">
      <div className="runhead-top">
        <span className={`status ${data.phase}`}>{live && <span className="pulse" />}{LABEL[data.phase]}</span>
        <span className="muted mono small">{s.swarmRunId}</span>
        <span className="muted">· {fmtDuration(elapsed)}</span>
        {s.status && !['running', 'cancelling'].includes(s.status) && s.status !== 'finished' && <span className="muted">· {s.status}</span>}
      </div>
      <p className="objective">{s.objective}</p>
      {total > 0 && (
        <div className="progress" title={`${read} of ${total} rows read`}>
          <div className="bar"><div style={{ width: `${(100 * read) / total}%` }} /></div>
          <span className="small muted">{read.toLocaleString()} of {total.toLocaleString()} rows read ({((100 * read) / total).toFixed(0)}%)</span>
        </div>
      )}
      <div className="tiles">{tiles.map(([k, v]) => <div key={k} className={`tile${k === 'Errors' && v ? ' bad' : ''}`}><b>{v}</b><span>{k}</span></div>)}</div>
    </header>
  )
}
