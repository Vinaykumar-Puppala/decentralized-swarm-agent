import { useRef, useState } from 'react'
import { uploadDatasets } from '../api'
import type { DatasetInfo } from '../types'

interface Props { dataset: DatasetInfo | null; onDataset: (d: DatasetInfo | null) => void; locked: boolean }

export default function DataPanel({ dataset, onDataset, locked }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [over, setOver] = useState(false)

  async function take(list: FileList | File[]) {
    const files = Array.from(list)
    if (!files.length) return
    setBusy(true); setError('')
    try { onDataset(await uploadDatasets(files)) } catch (e) { setError(e instanceof Error ? e.message : String(e)) } finally { setBusy(false) }
  }

  return (
    <section className="panel">
      <h2>Data <span className="muted">optional</span></h2>
      <div className={`drop${over ? ' over' : ''}`} role="button" tabIndex={0}
           onClick={() => input.current?.click()} onKeyDown={e => e.key === 'Enter' && input.current?.click()}
           onDragOver={e => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)}
           onDrop={e => { e.preventDefault(); setOver(false); void take(e.dataTransfer.files) }}>
        {busy ? 'Reading files…' : <>Drop CSV, Excel or Parquet files here<br /><span className="muted">or click to choose several</span></>}
        <input ref={input} type="file" multiple hidden accept=".csv,.xlsx,.xlsm,.xls,.parquet"
               onChange={e => { void take(e.target.files ?? []); e.target.value = '' }} />
      </div>
      {error && <p className="note bad">{error}</p>}
      {dataset && (
        <div className="tables">
          <p className="muted small">Each file and each Excel sheet is its own table. Tables are not joined.</p>
          {dataset.tables.map(t => (
            <details key={t.name}>
              <summary><span className="mono">{t.name}</span> <span className="muted">{t.rows.toLocaleString()} × {t.columns.length}</span></summary>
              <p className="small muted">{t.columns.join(', ')}</p>
              {t.cleaning.length > 0 && <ul className="small">{t.cleaning.map(c => <li key={c}>{c}</li>)}</ul>}
            </details>
          ))}
          <button className="link" disabled={locked} onClick={() => onDataset(null)}>Remove data</button>
        </div>
      )}
    </section>
  )
}
