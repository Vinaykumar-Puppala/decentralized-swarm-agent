import type { FeedItem } from '../types'
import { agentColor, fmtTime } from '../util'

type Board = Extract<FeedItem, { kind: 'board' }>
type Artifact = Extract<FeedItem, { kind: 'artifact' }>

export function BoardView({ feed }: { feed: FeedItem[] }) {
  const posts = feed.filter((f): f is Board => f.kind === 'board').slice().reverse()
  if (!posts.length) return <p className="empty">Nothing on the shared board yet.</p>
  return (
    <div className="stack">
      {posts.map(p => (
        <div key={p.id} className="item card">
          <time className="muted mono">{fmtTime(p.ts)}</time>
          <span className="who" style={{ color: agentColor(p.agent) }}><span className="swatch" style={{ background: agentColor(p.agent) }} />{p.agent}</span>
          <span className="text">{p.text}</span>
        </div>
      ))}
    </div>
  )
}

export function ArtifactsView({ feed }: { feed: FeedItem[] }) {
  const items = feed.filter((f): f is Artifact => f.kind === 'artifact').slice().reverse()
  if (!items.length) return <p className="empty">No artifacts published yet.</p>
  return (
    <div className="stack">
      {items.map(a => (
        <details key={a.id} className="item card block">
          <summary>
            <time className="muted mono">{fmtTime(a.ts)}</time>
            <span className="who" style={{ color: agentColor(a.agent) }}><span className="swatch" style={{ background: agentColor(a.agent) }} />{a.agent}</span>
            <span className={`pill k-${a.artifactKind}`}>{a.artifactKind.replace('_', ' ')}</span>
            <b>#{a.artifactId}</b> {a.name}
          </summary>
          <pre>{a.text}</pre>
        </details>
      ))}
    </div>
  )
}
