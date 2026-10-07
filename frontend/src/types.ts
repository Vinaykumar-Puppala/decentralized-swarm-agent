// Shared state the backend maintains and syncs with STATE_SNAPSHOT / STATE_DELTA.
export interface AgentState {
  state: string
  steps: number
  errors: number
  rowsRead: number
  lastTs: number
  action: string
  summary: string
}

export interface Coverage {
  total: number
  read: number
  ranges: [number, number][]
  agents: Record<string, number>
}

export interface SwarmState {
  swarmRunId: string
  status: string
  objective: string
  startedAt: number
  config: {
    llm: Record<string, unknown>
    nAgents?: number
    steps?: number
    dataset?: string | null
    tables: Record<string, number>
  }
  counts: { board: number; findings: number; finals: number; queries: number; row_reads: number; cross_reads: number; errors: number; events: number }
  agents: Record<string, AgentState>
  coverage: Record<string, Coverage>
}

// What the UI builds from the AG-UI event stream.
export type FeedItem =
  | { kind: 'message'; id: string; agent: string; ts: number; text: string; streaming: boolean; at: number }
  | { kind: 'tool'; id: string; agent: string; ts: number; name: string; args: string; result?: string; isError?: boolean }
  | { kind: 'board'; id: string; agent: string; ts: number; text: string }
  | { kind: 'artifact'; id: string; agent: string; ts: number; artifactId: number; name: string; artifactKind: string; text: string }
  | { kind: 'error'; id: string; agent: string; ts: number; text: string; errorKind: string }
  | { kind: 'access'; id: string; agent: string; ts: number; author: string; artifactId: number; cross: boolean }
  | { kind: 'lifecycle'; id: string; agent: string; ts: number; phase: 'started' | 'finished' | 'failed' | 'stopped'; text?: string }

export type Phase = 'idle' | 'connecting' | 'running' | 'finished' | 'cancelled' | 'error'

export interface RunData {
  phase: Phase
  error?: string
  feed: FeedItem[]
  shared: SwarmState | null
  steps: Record<string, string>      // agent -> current step name (from STEP_STARTED / STEP_FINISHED)
  result?: { status?: string; report?: string | null }
  eventCount: number
}

export interface LLMForm {
  provider: string
  model: string
  api_key: string
  base_url: string
  temperature: number | null
}

export interface RunForm {
  objective: string
  nAgents: number
  steps: number
  report: boolean
}

export interface TableInfo { name: string; rows: number; columns: string[]; cleaning: string[] }
export interface DatasetInfo { dataset_id: string; files: string[]; profile: string; tables: TableInfo[] }
export interface RunListItem { run_id: string; started: number; objective: string; status: string; active: boolean }
export interface AppConfig {
  providers: string[]
  default_base_urls: Record<string, string>
  defaults: { provider: string; model: string; base_url: string; temperature: number | null }
  key_in_env: Record<string, boolean>
}
