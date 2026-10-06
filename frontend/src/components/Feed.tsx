import { useMemo, useState } from 'react'
import type { FeedItem } from '../types'
import { agentColor, clip, fmtTime, prettyJson } from '../util'

function Who({ agent }: { agent: string }) {
  return <span className="who" style={{ color: agentColor(agent) }}><span className="swatch" style={{ background: agentColor(agent) }} />{agent}</span>
}

function Item({ it }: { it: FeedItem }) {
  const time = <time className="muted mono">{fmtTime(it.ts)}</time>
  switch (it.kind) {
    case 'message':
      if (it.agent === 'reporter')
        return <div className="item quiet">{time}<Who agent="reporter" /> wrote the post-hoc report{it.streaming ? '…' : ` (${it.text.length.toLocaleString()} characters, see the Report tab)`}</div>
      return <div className="item">{time}<Who agent={it.agent} /><span className="text">{it.text}{it.streaming && <span className="caret" />}</span></div>
    case 'tool':
      return (
        <div className={`item tool${it.isError ? ' err' : ''}`}>
          {time}<Who agent={it.agent} /><span className="pill mono">{it.name}</span>
          <div className="detail">
            <details><summary className="mono small">{clip(it.args.replace(/\s+/g, ' '), 140)}</summary><pre>{prettyJson(it.args)}</pre></details>
            {it.result === undefined
              ? <span className="muted small">waiting for result…</span>
              : <details><summary className="small">{it.isError ? 'failed' : 'result'}: {clip(it.result.replace(/\s+/g, ' '), 110)}</summary><pre>{it.result}</pre></details>}
          </div>
        </div>
      )
    case 'board':
      return <div className="item card">{time}<Who agent={it.agent} /><span className="pill">board</span><span className="text">{it.text}</span></div>
    case 'artifact':
      return (
        <div className="item card">
          {time}<Who agent={it.agent} /><span className={`pill k-${it.artifactKind}`}>{it.artifactKind.replace('_', ' ')}</span>
          <div className="detail"><details><summary><b>#{it.artifactId}</b> {clip(it.name, 90)}</summary><pre>{it.text}</pre></details></div>
        </div>
      )
    case 'error':
      return <div className="item err">{time}<Who agent={it.agent} /><span className="pill bad">{it.errorKind} error</span><span className="text mono small">{clip(it.text, 400)}</span></div>
    case 'access':
      return <div className="item quiet">{time}<Who agent={it.agent} /> read artifact #{it.artifactId} by {it.author}{it.cross ? ' (another agent)' : ''}</div>
    case 'lifecycle':
      return <div className="item quiet">{time}<Who agent={it.agent} /> {it.phase}{it.text ? ` · ${clip(it.text, 160)}` : ''}</div>
  }
}

export default function Feed({ feed, agents }: { feed: FeedItem[]; agents: string[] }) {
  const [agent, setAgent] = useState('all')
  const [quiet, setQuiet] = useState(true)
  const items = useMemo(() => {
    const keep = feed.filter(f => (agent === 'all' || f.agent === agent) && (quiet || (f.kind !== 'lifecycle' && f.kind !== 'access')))
    return keep.slice(-300).reverse()
  }, [feed, agent, quiet])

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
      {items.length === 0 ? <p className="empty">Waiting for the first agent event…</p> : <div className="feed">{items.map(it => <Item key={it.id} it={it} />)}</div>}
    </div>
  )
}
