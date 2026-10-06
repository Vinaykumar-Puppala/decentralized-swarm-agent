import { useState } from 'react'
import { testLLM } from '../api'
import type { AppConfig, LLMForm } from '../types'

interface Props { cfg: AppConfig | null; value: LLMForm; onChange: (v: LLMForm) => void }

const PLACEHOLDER: Record<string, string> = { openai: 'gpt-4o-mini', anthropic: 'claude-sonnet-4-5', litellm: 'my-model-alias', local: 'qwen2.5:3b' }

export default function ModelPanel({ cfg, value, onChange }: Props) {
  const [test, setTest] = useState<{ state: 'idle' | 'busy' | 'ok' | 'fail'; text?: string }>({ state: 'idle' })
  const set = (patch: Partial<LLMForm>) => onChange({ ...value, ...patch })
  const keyInEnv = cfg?.key_in_env[value.provider] || cfg?.key_in_env.generic

  async function runTest() {
    setTest({ state: 'busy' })
    try {
      const r = await testLLM(value)
      setTest(r.ok ? { state: 'ok', text: `Connected in ${r.seconds}s` } : { state: 'fail', text: r.error })
    } catch (e) { setTest({ state: 'fail', text: String(e) }) }
  }

  return (
    <section className="panel">
      <h2>Model</h2>
      <label>Provider
        <select value={value.provider} onChange={e => set({ provider: e.target.value })}>
          {(cfg?.providers ?? ['openai', 'anthropic', 'litellm', 'local']).map(p => <option key={p}>{p}</option>)}
        </select>
      </label>
      <label>Model name
        <input value={value.model} placeholder={PLACEHOLDER[value.provider]} onChange={e => set({ model: e.target.value })} />
      </label>
      <label>Base URL <span className="muted">(blank = default)</span>
        <input value={value.base_url} placeholder={cfg?.default_base_urls[value.provider] ?? ''} onChange={e => set({ base_url: e.target.value })} />
      </label>
      <label>API key
        <input type="password" autoComplete="off" value={value.api_key}
               placeholder={keyInEnv ? 'using the key from the server .env' : value.provider === 'local' ? 'not needed for local servers' : 'required'}
               onChange={e => set({ api_key: e.target.value })} />
      </label>
      <label className="row">
        <input type="checkbox" checked={value.temperature !== null} onChange={e => set({ temperature: e.target.checked ? 0.7 : null })} />
        Send temperature
        <input type="range" min={0} max={1.5} step={0.1} disabled={value.temperature === null} value={value.temperature ?? 0.7}
               onChange={e => set({ temperature: Number(e.target.value) })} />
        <span className="mono">{value.temperature ?? 'off'}</span>
      </label>
      <button className="secondary" onClick={runTest} disabled={test.state === 'busy'}>{test.state === 'busy' ? 'Testing…' : 'Test connection'}</button>
      {test.state === 'ok' && <p className="note ok">{test.text}</p>}
      {test.state === 'fail' && <p className="note bad">{test.text}</p>}
    </section>
  )
}
