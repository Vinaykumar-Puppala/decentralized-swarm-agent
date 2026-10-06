import { useEffect, useState } from 'react'

export const agentColor = (name: string) =>
  name === 'reporter' || name === 'swarm' ? 'var(--muted)' : `hsl(${(parseInt(name.replace(/\D/g, '') || '0', 10) * 57 + 220) % 360} 62% 56%)`

export const fmtTime = (ts: number) => new Date(ts * 1000).toLocaleTimeString([], { hour12: false })

export function fmtDuration(seconds: number) {
  const s = Math.max(0, Math.round(seconds))
  return s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m` : s >= 60 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${s}s`
}

export const clip = (s: string, n: number) => (s.length > n ? s.slice(0, n) + '…' : s)

export function prettyJson(text: string) {
  try { return JSON.stringify(JSON.parse(text), null, 2) } catch { return text }
}

/** Re-renders every `ms` while `active`, returns the current time in seconds. */
export function useNow(active: boolean, ms = 1000) {
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (!active) return
    const id = window.setInterval(() => setNow(Date.now() / 1000), ms)
    return () => window.clearInterval(id)
  }, [active, ms])
  return now
}

export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try { const raw = localStorage.getItem(key); return raw ? { ...initial, ...JSON.parse(raw) } : initial } catch { return initial }
  })
  useEffect(() => { try { localStorage.setItem(key, JSON.stringify(value)) } catch { /* private mode */ } }, [key, value])
  return [value, setValue] as const
}
