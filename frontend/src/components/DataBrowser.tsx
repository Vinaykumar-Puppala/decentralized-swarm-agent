import { useEffect, useState } from 'react'
import { datasetRows, type RowsPage } from '../api'
import type { DatasetInfo, SwarmState } from '../types'
import CoverageBar from './CoverageBar'

const PAGE = 25

function inRanges(id: number, ranges: [number, number][]) {
  return ranges.some(([a, b]) => id >= a && id <= b)
}

export default function DataBrowser({ dataset, shared }: { dataset: DatasetInfo | null; shared: SwarmState | null }) {
  const tables = dataset?.tables.map(t => t.name) ?? []
  const [table, setTable] = useState('')
  const [offset, setOffset] = useState(0)
  const [page, setPage] = useState<RowsPage | null>(null)
  const [error, setError] = useState('')
  const current = tables.includes(table) ? table : tables[0] ?? ''

  useEffect(() => { setOffset(0) }, [current, dataset?.dataset_id])
  useEffect(() => {
    if (!dataset || !current) { setPage(null); return }
    let live = true
    datasetRows(dataset.dataset_id, current, offset, PAGE).then(p => { if (live) { setPage(p); setError('') } }).catch(e => { if (live) setError(String(e)) })
    return () => { live = false }
  }, [dataset, current, offset])

  const cov = shared?.coverage ?? {}
  const readRanges = cov[current]?.ranges ?? []

  return (
    <div>
      {Object.keys(cov).length > 0 && (
        <div className="stack">
          <p className="muted small">Row coverage: every segment is a block of rows that at least one agent has read with data_rows.</p>
          {Object.entries(cov).map(([n, c]) => <CoverageBar key={n} name={n} cov={c} />)}
        </div>
      )}
      {!dataset ? <p className="empty">Upload data in the sidebar to browse it here. Rows the agents have read are marked once a run is going.</p> : (
        <>
          <div className="toolbar">
            <div className="chips">{tables.map(t => <button key={t} className={t === current ? 'chip on' : 'chip'} onClick={() => setTable(t)}>{t}</button>)}</div>
            {page && (
              <div className="pager small">
                <button className="secondary" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>Previous</button>
                <span className="muted">rows {offset + 1}–{Math.min(offset + PAGE, page.total)} of {page.total.toLocaleString()}</span>
                <button className="secondary" disabled={offset + PAGE >= page.total} onClick={() => setOffset(offset + PAGE)}>Next</button>
              </div>
            )}
          </div>
          {error && <p className="note bad">{error}</p>}
          {page && (
            <div className="tablewrap">
              <table>
                <thead><tr><th>read</th><th>_row</th>{page.columns.map(c => <th key={c}>{c}</th>)}</tr></thead>
                <tbody>
                  {page.rows.map((r, i) => {
                    const id = page.row_ids[i]
                    const seen = inRanges(id, readRanges)
                    return (
                      <tr key={id} className={seen ? 'seen' : ''}>
                        <td title={seen ? 'read by an agent' : 'not read yet'}>{seen ? '●' : ''}</td>
                        <td className="mono muted">{id}</td>
                        {r.map((v, j) => <td key={j}>{v === null ? <span className="muted">null</span> : String(v)}</td>)}
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          )}
        </>
      )}
    </div>
  )
}
