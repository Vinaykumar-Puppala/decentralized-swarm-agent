import type { Coverage } from '../types'

/** One horizontal bar per table; every contiguous range of rows read by the swarm is drawn as a segment. */
export default function CoverageBar({ name, cov }: { name: string; cov: Coverage }) {
  const pct = cov.total ? (100 * cov.read) / cov.total : 0
  return (
    <div className="coverage">
      <div className="coverage-head">
        <span className="mono">{name}</span>
        <span className="muted">{cov.read.toLocaleString()} / {cov.total.toLocaleString()} rows read · {pct.toFixed(0)}%</span>
      </div>
      <svg className="coverage-bar" viewBox="0 0 1000 14" preserveAspectRatio="none" role="img" aria-label={`${pct.toFixed(0)}% of ${name} read`}>
        <rect x="0" y="0" width="1000" height="14" className="coverage-track" />
        {cov.ranges.map(([a, b]) => (
          <rect key={a} x={(a / cov.total) * 1000} y="0" width={Math.max(((b - a + 1) / cov.total) * 1000, 2)} height="14" className="coverage-fill" />
        ))}
      </svg>
    </div>
  )
}
