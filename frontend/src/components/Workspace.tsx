import { useMemo, useState } from 'react'
import type { FeedItem } from '../types'
import { agentColor, fmtTime } from '../util'
import { Tag, Who, artifactLabel, artifactTone } from './bits'

type Board = Extract<FeedItem, { kind: 'board' }>
type Artifact = Extract<FeedItem, { kind: 'artifact' }>

export function BoardView({ feed }: { feed: FeedItem[] }) {
  const [agent, setAgent] = useState('all')
  const [q, setQ] = useState('')
  const all = useMemo(() => feed.filter((f): f is Board => f.kind === 'board'), [feed])      // every post, none dropped
  const names = useMemo(() => Array.from(new Set(all.map(p => p.agent))).sort(), [all])
  const posts = useMemo(() => all.filter(p => (agent === 'all' || p.agent === agent) && (!q || p.text.toLowerCase().includes(q.toLowerCase()))).slice().reverse(), [all, agent, q])
  if (!all.length) return <p className="empty">Nothing on the shared board yet.</p>
  return (
    <div>
      <div className="toolbar">
        <div className="chips">
          <button className={agent === 'all' ? 'chip on' : 'chip'} onClick={() => setAgent('all')}>All <span className="count">{all.length}</span></button>
          {names.map(a => (
            <button key={a} className={agent === a ? 'chip on' : 'chip'} onClick={() => setAgent(a)}>
              <span className="swatch" style={{ background: agentColor(a) }} />{a} <span className="count">{all.filter(p => p.agent === a).length}</span>
            </button>
          ))}
        </div>
        <input className="search" type="search" placeholder="Search the board…" value={q} onChange={e => setQ(e.target.value)} aria-label="Search board messages" />
      </div>
      <p className="muted small">All {all.length} board messages, newest first{posts.length !== all.length ? ` · showing ${posts.length}` : ''}.</p>
      <div className="feed">
        {posts.map(p => (
          <article key={p.id} className="item" style={{ '--agent': agentColor(p.agent) } as React.CSSProperties}>
            <div className="item-head"><Who agent={p.agent} /><Tag tone="accent">board</Tag><time className="muted mono">{fmtTime(p.ts)}</time></div>
            <div className="item-body text">{p.text}</div>
          </article>
        ))}
        {!posts.length && <p className="empty">No board messages match.</p>}
      </div>
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
