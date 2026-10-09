export function genId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export function formatTime(ts: number): string {
  const d = new Date(ts)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(
    d.getMinutes(),
  )}`
}

export function truncate(text: string, max = 20): string {
  const normalized = text.trim().replace(/\s+/g, ' ')
  return normalized.length > max ? `${normalized.slice(0, max)}…` : normalized
}

export function percent(value: number): string {
  const safe = Number.isFinite(value) ? value : 0
  return `${(safe * 100).toFixed(2)}%`
}
