import type { FeedItem } from '../types'
import { agentColor, fmtTime } from '../util'
import { Tag, Who, artifactLabel, artifactTone } from './bits'

type Board = Extract<FeedItem, { kind: 'board' }>
type Artifact = Extract<FeedItem, { kind: 'artifact' }>

export function BoardView({ feed }: { feed: FeedItem[] }) {
  const posts = feed.filter((f): f is Board => f.kind === 'board').slice().reverse()
  if (!posts.length) return <p className="empty">Nothing on the shared board yet.</p>
  return (
    <div className="feed">
      {posts.map(p => (
        <article key={p.id} className="item" style={{ '--agent': agentColor(p.agent) } as React.CSSProperties}>
          <div className="item-head"><Who agent={p.agent} /><Tag tone="accent">board</Tag><time className="muted mono">{fmtTime(p.ts)}</time></div>
          <div className="item-body text">{p.text}</div>
        </article>
      ))}
    </div>
  )
}

export function ArtifactsView({ feed }: { feed: FeedItem[] }) {
  const items = feed.filter((f): f is Artifact => f.kind === 'artifact').slice().reverse()
  if (!items.length) return <p className="empty">No artifacts published yet.</p>
  return (
    <div className="feed">
      {items.map(a => (
        <article key={a.id} className="item" style={{ '--agent': agentColor(a.agent) } as React.CSSProperties}>
          <div className="item-head"><Who agent={a.agent} /><Tag tone={artifactTone(a.artifactKind)}>{artifactLabel(a.artifactKind)}</Tag><time className="muted mono">{fmtTime(a.ts)}</time></div>
          <div className="item-body"><details><summary><b>#{a.artifactId}</b> {a.name}</summary><pre>{a.text}</pre></details></div>
        </article>
      ))}
    </div>
  )
}
