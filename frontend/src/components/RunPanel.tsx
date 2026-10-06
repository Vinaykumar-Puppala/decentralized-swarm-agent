import type { RunForm } from '../types'

interface Props {
  value: RunForm
  onChange: (v: RunForm) => void
  busy: boolean
  canStart: boolean
  hint?: string
  onStart: () => void
  onStop: () => void
  stopping: boolean
}

export default function RunPanel({ value, onChange, busy, canStart, hint, onStart, onStop, stopping }: Props) {
  const set = (patch: Partial<RunForm>) => onChange({ ...value, ...patch })
  return (
    <section className="panel">
      <h2>Run</h2>
      <label>Common objective
        <textarea rows={4} value={value.objective} onChange={e => set({ objective: e.target.value })} />
      </label>
      <label className="row">Agents <input type="range" min={1} max={8} value={value.nAgents} onChange={e => set({ nAgents: Number(e.target.value) })} /><span className="mono">{value.nAgents}</span></label>
      <label className="row" title="Each data_rows call reads up to 100 rows, so more steps means more of the data is read.">
        Max steps <input type="range" min={1} max={40} value={value.steps} onChange={e => set({ steps: Number(e.target.value) })} /><span className="mono">{value.steps}</span>
      </label>
      <label className="row"><input type="checkbox" checked={value.report} onChange={e => set({ report: e.target.checked })} />Write a post-hoc report when done</label>
      <div className="actions">
        <button className="primary" disabled={!canStart || busy} onClick={onStart}>{busy ? 'Running…' : `Run ${value.nAgents} agents`}</button>
        <button className="danger" disabled={!busy || stopping} onClick={onStop}>{stopping ? 'Stopping…' : 'Stop'}</button>
      </div>
      {hint && <p className="note muted">{hint}</p>}
    </section>
  )
}
