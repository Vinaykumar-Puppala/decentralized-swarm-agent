import { useCallback, useEffect, useRef, useState } from 'react'
import { HttpAgent, type AgentSubscriber } from '@ag-ui/client'
import type { FeedItem, LLMForm, RunData, RunForm, SwarmState } from '../types'

// The AG-UI event fields we read. The stream carries more; unknown event types are ignored.
type Ev = { type: string; timestamp?: number; subagentRunId?: string } & Record<string, any>

const uuid = () => (globalThis.crypto?.randomUUID?.() ?? `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`)
const agentOf = (e: Ev): string => e.name ?? (e.subagentRunId ? String(e.subagentRunId).split(':').pop()! : 'swarm')
const tsOf = (e: Ev) => (e.timestamp ? e.timestamp / 1000 : Date.now() / 1000)

const empty = (phase: RunData['phase'] = 'idle'): RunData => ({ phase, feed: [], shared: null, steps: {}, eventCount: 0 })

/** Fold a batch of AG-UI events into the run data. Pure: copies what it changes, so React sees new references. */
export function applyEvents(prev: RunData, events: Ev[]): RunData {
  const s: RunData = { ...prev, feed: prev.feed.slice(), steps: { ...prev.steps }, eventCount: prev.eventCount + events.length }
  const at = new Map<string, number>()
  s.feed.forEach((f, i) => at.set(f.id, i))
  const put = (item: FeedItem) => {
    const i = at.get(item.id)
    if (i === undefined) { at.set(item.id, s.feed.length); s.feed.push(item) } else s.feed[i] = item
  }
  const edit = (id: string, fn: (f: FeedItem) => FeedItem) => {
    const i = at.get(id)
    if (i !== undefined) s.feed[i] = fn(s.feed[i])
  }

  for (const e of events) {
    const ts = tsOf(e)
    switch (e.type) {
      case 'RUN_STARTED': s.phase = 'running'; break
      case 'RUN_FINISHED':
        s.phase = e.outcome?.type === 'cancelled' ? 'cancelled' : 'finished'
        s.result = e.result ?? undefined
        break
      case 'RUN_ERROR': s.phase = 'error'; s.error = e.message; break

      case 'SUBAGENT_STARTED':
        put({ kind: 'lifecycle', id: `life-${e.subagentRunId}-start`, agent: e.name, ts, phase: 'started', text: e.description })
        break
      case 'SUBAGENT_FINISHED':
        put({ kind: 'lifecycle', id: `life-${e.subagentRunId}-end`, agent: agentOf(e), ts, phase: 'finished',
              text: e.result?.steps != null ? `${e.result.steps} steps` : undefined })
        break
      case 'SUBAGENT_ERROR':
        put({ kind: 'lifecycle', id: `life-${e.subagentRunId}-end`, agent: agentOf(e), ts, phase: 'failed', text: e.message })
        break
      case 'STEP_STARTED': s.steps[agentOf(e)] = e.stepName; break
      case 'STEP_FINISHED': if (s.steps[agentOf(e)] === e.stepName) delete s.steps[agentOf(e)]; break

      case 'TEXT_MESSAGE_START':
        put({ kind: 'message', id: e.messageId, agent: e.name ?? agentOf(e), ts, text: '', streaming: true })
        break
      case 'TEXT_MESSAGE_CONTENT':
        edit(e.messageId, f => (f.kind === 'message' ? { ...f, text: f.text + e.delta } : f))
        break
      case 'TEXT_MESSAGE_END':
        edit(e.messageId, f => (f.kind === 'message' ? { ...f, streaming: false } : f))
        break

      case 'TOOL_CALL_START':
        put({ kind: 'tool', id: e.toolCallId, agent: agentOf(e), ts, name: e.toolCallName, args: '' })
        break
      case 'TOOL_CALL_ARGS':
        edit(e.toolCallId, f => (f.kind === 'tool' ? { ...f, args: f.args + e.delta } : f))
        break
      case 'TOOL_CALL_RESULT':
        edit(e.toolCallId, f => (f.kind === 'tool' ? { ...f, result: e.content, isError: String(e.content).startsWith('ERROR:') } : f))
        break

      case 'ACTIVITY_SNAPSHOT':
        if (e.activityType === 'board_post')
          put({ kind: 'board', id: e.messageId, agent: e.content.agent, ts, text: e.content.content })
        else if (e.activityType === 'artifact')
          put({ kind: 'artifact', id: e.messageId, agent: e.content.agent, ts, artifactId: e.content.id, name: e.content.name,
                artifactKind: e.content.kind, text: e.content.content })
        break

      case 'CUSTOM':
        if (e.name === 'swarm.agent_error')
          put({ kind: 'error', id: `err-${s.feed.length}-${ts}`, agent: e.value.agent, ts, text: e.value.error, errorKind: e.value.kind })
        else if (e.name === 'swarm.access')
          put({ kind: 'access', id: `acc-${s.feed.length}-${ts}`, agent: e.value.reader, ts, author: e.value.author,
                artifactId: e.value.artifact_id, cross: !!e.value.cross_agent })
        else if (e.name === 'swarm.agent_cancelled')
          put({ kind: 'lifecycle', id: `stop-${e.value.agent}`, agent: e.value.agent, ts, phase: 'stopped' })
        break
    }
  }
  return s
}

export interface SwarmRun {
  data: RunData
  start: (llm: LLMForm, run: RunForm, datasetId: string | null) => Promise<void>
  attach: (swarmRunId: string) => Promise<void>
  clear: () => void
  busy: boolean
}

export function useSwarmRun(onSettled?: () => void): SwarmRun {
  const [data, setData] = useState<RunData>(empty())
  const agentRef = useRef<HttpAgent | null>(null)
  const buffer = useRef<Ev[]>([])
  const latestState = useRef<SwarmState | null>(null)
  const timer = useRef<number | null>(null)
  const generation = useRef(0)         // ignores late callbacks from a run the user has already left

  const flush = useCallback(() => {
    timer.current = null
    const batch = buffer.current
    buffer.current = []
    setData(prev => {
      const next = batch.length ? applyEvents(prev, batch) : { ...prev }
      next.shared = latestState.current ?? next.shared
      return next
    })
  }, [])
  const schedule = useCallback(() => { if (timer.current === null) timer.current = window.setTimeout(flush, 80) }, [flush])

  const run = useCallback(async (forwardedProps: Record<string, unknown>, objective: string | null) => {
    agentRef.current?.abortRun()                        // leaving a run only detaches the view: the swarm keeps going server-side
    const gen = ++generation.current
    buffer.current = []
    latestState.current = null
    setData(empty('connecting'))

    const agent = new HttpAgent({ url: '/agui', threadId: uuid() })
    if (objective) agent.addMessage({ id: uuid(), role: 'user', content: objective })
    agentRef.current = agent

    const subscriber: AgentSubscriber = {
      onEvent: ({ event }) => { if (gen === generation.current) { buffer.current.push(event as Ev); schedule() } },
      onStateChanged: ({ state }) => { if (gen === generation.current) { latestState.current = state as SwarmState; schedule() } },
      onRunFailed: ({ error }) => {
        if (gen !== generation.current) return
        buffer.current.push({ type: 'RUN_ERROR', message: error?.message ?? String(error) }); schedule()
      },
    }
    try {
      await agent.runAgent({ runId: uuid(), forwardedProps }, subscriber)
    } catch (err) {
      if (gen === generation.current && !(err instanceof DOMException && err.name === 'AbortError')) {
        buffer.current.push({ type: 'RUN_ERROR', message: err instanceof Error ? err.message : String(err) })
      }
    } finally {
      if (gen === generation.current) { if (timer.current) window.clearTimeout(timer.current); flush(); onSettled?.() }
    }
  }, [flush, schedule, onSettled])

  const start = useCallback((llm: LLMForm, form: RunForm, datasetId: string | null) =>
    run({ llm, nAgents: form.nAgents, steps: form.steps, report: form.report, datasetId }, form.objective), [run])
  const attach = useCallback((swarmRunId: string) => run({ attachRunId: swarmRunId }, null), [run])
  const clear = useCallback(() => { generation.current++; agentRef.current?.abortRun(); setData(empty()) }, [])

  useEffect(() => () => { generation.current++; agentRef.current?.abortRun() }, [])
  return { data, start, attach, clear, busy: data.phase === 'connecting' || data.phase === 'running' }
}
