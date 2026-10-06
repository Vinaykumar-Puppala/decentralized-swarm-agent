import type { FeedItem, RunData } from '../types'
import { agentColor } from '../util'
import Markdown from './Markdown'

export default function Report({ data }: { data: RunData }) {
  const reportMsg = data.feed.find((f): f is Extract<FeedItem, { kind: 'message' }> => f.kind === 'message' && f.agent === 'reporter')
  const report = data.result?.report || reportMsg?.text || ''
  const finals = data.feed.filter((f): f is Extract<FeedItem, { kind: 'artifact' }> => f.kind === 'artifact' && f.artifactKind === 'final_answer')
  const running = data.phase === 'running' || data.phase === 'connecting'

  return (
    <div className="stack">
      {report
        ? <section className="paper"><h3>Post-hoc report <span className="muted small">by an observer, not an agent{reportMsg?.streaming ? ' · writing…' : ''}</span></h3><Markdown>{report}</Markdown></section>
        : <p className="empty">{running ? 'The report appears here when every agent has finished.' : 'No report was written for this run.'}</p>}
      {finals.length > 0 && <h3>Final answers from the agents</h3>}
      {finals.map(f => (
        <details key={f.id} className="item card block">
          <summary><span className="who" style={{ color: agentColor(f.agent) }}><span className="swatch" style={{ background: agentColor(f.agent) }} />{f.agent}</span> {f.name}</summary>
          <Markdown>{f.text}</Markdown>
        </details>
      ))}
    </div>
  )
}
