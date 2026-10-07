import type { AppConfig, DatasetInfo, LLMForm, LlmCallDetail, RunListItem } from './types'

async function json<T>(r: Response): Promise<T> {
  if (!r.ok) {
    let detail = r.statusText
    try { detail = (await r.json()).detail ?? detail } catch { /* not json */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return r.json() as Promise<T>
}

export const getConfig = () => fetch('/api/config').then(r => json<AppConfig>(r))
export const listRuns = () => fetch('/api/runs').then(r => json<RunListItem[]>(r))
export const stopRun = (id: string) => fetch(`/api/runs/${id}/stop`, { method: 'POST' }).then(r => json<{ ok: boolean }>(r))

export const testLLM = (llm: LLMForm) =>
  fetch('/api/llm/test', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(llm) })
    .then(r => json<{ ok: boolean; seconds?: number; reply?: string; error?: string }>(r))

export function uploadDatasets(files: File[]) {
  const body = new FormData()
  files.forEach(f => body.append('files', f, f.name))
  return fetch('/api/datasets', { method: 'POST', body }).then(r => json<DatasetInfo>(r))
}

export interface RowsPage { table: string; total: number; offset: number; columns: string[]; row_ids: number[]; rows: unknown[][] }
export const datasetRows = (id: string, table: string, offset: number, limit = 50) =>
  fetch(`/api/datasets/${id}/rows?` + new URLSearchParams({ table, offset: String(offset), limit: String(limit) }))
    .then(r => json<RowsPage>(r))

export const llmCall = (runId: string, id: number) => fetch(`/api/runs/${runId}/llm-calls/${id}`).then(r => json<LlmCallDetail>(r))
