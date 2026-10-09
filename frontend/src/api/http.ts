/** 极简 fetch 封装：统一 base URL 与 `{"error": ...}` 错误体。 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

async function errorMessage(resp: Response): Promise<string> {
  try {
    const payload: unknown = await resp.json()
    if (payload && typeof payload === 'object' && 'error' in payload) {
      const err = (payload as { error: unknown }).error
      if (typeof err === 'string') return err
    }
  } catch {
    // 非 JSON 错误体
  }
  return `请求失败（HTTP ${resp.status}）`
}

export async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const resp = await fetch(apiUrl(path), { signal, headers: { Accept: 'application/json' } })
  if (!resp.ok) throw new Error(await errorMessage(resp))
  return (await resp.json()) as T
}

export async function postJson<T>(
  path: string,
  body: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const resp = await fetch(apiUrl(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!resp.ok) throw new Error(await errorMessage(resp))
  return (await resp.json()) as T
}
