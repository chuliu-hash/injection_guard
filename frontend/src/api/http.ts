/** fetch 封装：统一 base URL、Bearer 鉴权、401 自动刷新重试与 `{"error": ...}` 错误体。 */

import { getAccessToken, notifyUnauthorized } from './authToken'

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/$/, '')

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`
}

/** 带 HTTP 状态码的错误，便于上层区分 401 / 404 等。 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
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

interface RequestOptions {
  method?: 'GET' | 'POST' | 'PATCH' | 'PUT' | 'DELETE'
  body?: unknown
  signal?: AbortSignal
  /** 是否附带 `Authorization`（默认 true）。 */
  auth?: boolean
  /** 401 时是否尝试刷新令牌并重试一次（默认 true）。 */
  retryOnUnauthorized?: boolean
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = 'GET', body, signal, auth = true, retryOnUnauthorized = true } = options

  const send = (): Promise<Response> => {
    const headers: Record<string, string> = { Accept: 'application/json' }
    if (body !== undefined) headers['Content-Type'] = 'application/json'
    const token = auth ? getAccessToken() : null
    if (token) headers.Authorization = `Bearer ${token}`
    return fetch(apiUrl(path), {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
      credentials: 'include', // 同源携带 refresh Cookie
    })
  }

  let resp = await send()
  if (resp.status === 401 && auth && retryOnUnauthorized) {
    const refreshed = await notifyUnauthorized()
    if (refreshed) resp = await send()
  }
  if (!resp.ok) throw new ApiError(await errorMessage(resp), resp.status)
  if (resp.status === 204) return undefined as T
  return (await resp.json()) as T
}

export function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { signal })
}

export function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'POST', body, signal })
}

export function patchJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'PATCH', body, signal })
}

export function putJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'PUT', body, signal })
}

export function deleteJson<T = void>(path: string, signal?: AbortSignal): Promise<T> {
  return request<T>(path, { method: 'DELETE', signal })
}
