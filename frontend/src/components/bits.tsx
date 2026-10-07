import { useEffect, useRef, useState } from 'react'
import { agentColor } from '../util'

export function Who({ agent }: { agent: string }) {
  const c = agentColor(agent)
  return <span className="who" style={{ color: c }}><span className="swatch" style={{ background: c }} />{agent}</span>
}

export type Tone = 'neutral' | 'accent' | 'ok' | 'info' | 'warn' | 'bad'

/** A compact label that only takes the width of its text. */
export function Tag({ tone = 'neutral', mono, children }: { tone?: Tone; mono?: boolean; children: React.ReactNode }) {
  return <span className={`tag tone-${tone}${mono ? ' mono' : ''}`}>{children}</span>
}

export const artifactTone = (kind: string): Tone =>
  kind === 'final_answer' ? 'ok' : kind === 'query_result' ? 'info' : kind === 'finding' ? 'accent' : 'neutral'

export const artifactLabel = (kind: string) => kind.replace('_', ' ')

/** Reveals new text progressively instead of dropping a whole paragraph in at once. */
export function Typed({ text, animate }: { text: string; animate: boolean }) {
  const [shown, setShown] = useState(animate ? 0 : text.length)
  const target = useRef(text.length)
  target.current = text.length
  useEffect(() => {
    if (!animate) { setShown(text.length); return }
    let raf = 0
    const tick = () => {
      setShown(n => {
        const left = target.current - n
        if (left <= 0) return n
        return n + Math.max(2, Math.ceil(left / 14))
      })
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [animate, text.length])
  const done = shown >= text.length
  return <>{done ? text : text.slice(0, shown)}{animate && !done && <span className="caret" />}</>
}
