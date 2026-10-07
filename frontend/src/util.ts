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

export type Theme = 'system' | 'light' | 'dark'

/** light / dark / follow-the-OS; the choice is saved and applied as data-theme on <html>. */
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    try { const t = localStorage.getItem('swarm.theme'); return t === 'light' || t === 'dark' ? t : 'system' } catch { return 'system' }
  })
  useEffect(() => {
    const el = document.documentElement
    if (theme === 'system') el.removeAttribute('data-theme'); else el.dataset.theme = theme
    try { if (theme === 'system') localStorage.removeItem('swarm.theme'); else localStorage.setItem('swarm.theme', theme) } catch { /* private mode */ }
  }, [theme])
  return [theme, setTheme] as const
}

export const fmtNum = (n: number) => n.toLocaleString()
export const fmtTokens = (n: number) => (n >= 1e6 ? `${(n / 1e6).toFixed(2)}M` : n >= 1e4 ? `${(n / 1e3).toFixed(1)}k` : n.toLocaleString())
export const fmtMs = (ms: number) => (ms >= 60000 ? `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s` : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`)
