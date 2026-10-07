import type { Theme } from '../util'

const ICON: Record<Theme, React.ReactNode> = {
  light: <path d="M12 4V2m0 20v-2M4 12H2m20 0h-2M5.6 5.6 4.2 4.2m15.6 15.6-1.4-1.4M18.4 5.6l1.4-1.4M4.2 19.8l1.4-1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z" />,
  dark: <path d="M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5Z" />,
  system: <path d="M4 5h16v11H4zM9 20h6M12 16v4" />,
}
const NEXT: Record<Theme, Theme> = { system: 'light', light: 'dark', dark: 'system' }

export default function ThemeToggle({ theme, onChange }: { theme: Theme; onChange: (t: Theme) => void }) {
  return (
    <button className="icon-btn" onClick={() => onChange(NEXT[theme])} title={`Theme: ${theme} (click for ${NEXT[theme]})`} aria-label={`Theme: ${theme}. Switch to ${NEXT[theme]}`}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden>{ICON[theme]}</svg>
    </button>
  )
}
