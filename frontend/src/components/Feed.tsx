import { useMemo, useState } from 'react'
import type { FeedItem } from '../types'
import { agentColor, clip, fmtTime, prettyJson } from '../util'
import { Tag, Typed, Who, artifactLabel, artifactTone } from './bits'

export function Item({ it, fresh }: { it: FeedItem; fresh?: boolean }) {
  const time = <time className="muted mono">{fmtTime(it.ts)}</time>
  const style = { '--agent': agentColor(it.agent) } as React.CSSProperties
  switch (it.kind) {
    case 'message':
      if (it.agent === 'reporter')
        return <div className="note-line">{time}<Who agent="reporter" /><span>wrote the post-hoc report{it.streaming ? '…' : ` (${it.text.length.toLocaleString()} characters, see the Report tab)`}</span></div>
      return (
        <article className="item" style={style}>
          <div className="item-head"><Who agent={it.agent} />{time}</div>
          <div className="item-body text"><Typed text={it.text} animate={!!fresh} /></div>
        </article>
      )
    case 'tool':
      return (
        <article className={`item${it.isError ? ' is-bad' : ''}`} style={style}>
          <div className="item-head">
            <Who agent={it.agent} /><Tag tone="info" mono>{it.name}</Tag>
            {it.result === undefined ? <Tag tone="warn">running</Tag> : it.isError ? <Tag tone="bad">failed</Tag> : <Tag tone="ok">done</Tag>}
            {time}
          </div>
          <div className="item-body">
            <details><summary className="mono small">{clip(it.args.replace(/\s+/g, ' '), 140)}</summary><pre>{prettyJson(it.args)}</pre></details>
            {it.result !== undefined && <details><summary className="small">{it.isError ? 'error' : 'result'}: {clip(it.result.replace(/^ERROR:\s*/, '').replace(/\s+/g, ' '), 110)}</summary><pre>{it.result}</pre></details>}
          </div>
        </article>
      )
    case 'board':
      return (
        <article className="item" style={style}>
          <div className="item-head"><Who agent={it.agent} /><Tag tone="accent">board</Tag>{time}</div>
          <div className="item-body text">{it.text}</div>
        </article>
      )
    case 'artifact':
      return (
        <article className="item" style={style}>
          <div className="item-head"><Who agent={it.agent} /><Tag tone={artifactTone(it.artifactKind)}>{artifactLabel(it.artifactKind)}</Tag>{time}</div>
          <div className="item-body"><details><summary><b>#{it.artifactId}</b> {clip(it.name, 100)}</summary><pre>{it.text}</pre></details></div>
        </article>
      )
    case 'error':
      return (
        <article className="item is-bad" style={style}>
          <div className="item-head"><Who agent={it.agent} /><Tag tone="bad">{it.errorKind} error</Tag>{time}</div>
          <div className="item-body text mono small">{clip(it.text, 400)}</div>
        </article>
      )
    case 'access':
      return <div className="note-line">{time}<Who agent={it.agent} /><span>read artifact #{it.artifactId} by {it.author}{it.cross ? ' (another agent)' : ''}</span></div>
    case 'lifecycle':
      return <div className="note-line">{time}<Who agent={it.agent} /><span>{it.phase}{it.text ? ` · ${clip(it.text, 160)}` : ''}</span></div>
  }
}

export default function Feed({ feed, agents, live }: { feed: FeedItem[]; agents: string[]; live: boolean }) {
  const [agent, setAgent] = useState('all')
  const [quiet, setQuiet] = useState(true)
  const items = useMemo(() => {
    const keep = feed.filter(f => (agent === 'all' || f.agent === agent) && (quiet || (f.kind !== 'lifecycle' && f.kind !== 'access')))
    return keep.slice(-300).reverse()
  }, [feed, agent, quiet])
  const freshIds = useMemo(() => {                      // only the newest few agent messages type themselves out
    const ids = new Set<string>()
    if (!live) return ids
    for (const it of items) { if (ids.size >= 3) break; if (it.kind === 'message' && it.agent !== 'reporter' && Date.now() - it.at < 6000) ids.add(it.id) }
    return ids
  }, [items, live])

  return (
    <div>
      <div className="toolbar">
        <div className="chips">
          <button className={agent === 'all' ? 'chip on' : 'chip'} onClick={() => setAgent('all')}>All</button>
          {agents.map(a => (
            <button key={a} className={agent === a ? 'chip on' : 'chip'} onClick={() => setAgent(a)}>
              <span className="swatch" style={{ background: agentColor(a) }} />{a}
            </button>
          ))}
        </div>
        <label className="row small"><input type="checkbox" checked={quiet} onChange={e => setQuiet(e.target.checked)} /> show reads and lifecycle</label>
      </div>
      <p className="muted small">Newest first. Agent messages are the audit summary of each decision (what it saw, what it did and why), not hidden chain-of-thought.</p>
      {items.length === 0 ? <p className="empty">Waiting for the first agent event…</p> : <div className="feed">{items.map(it => <Item key={it.id} it={it} fresh={freshIds.has(it.id)} />)}</div>}
    </div>
  )
}
