/**
 * 基于 `fetch` 的 POST-SSE 客户端。
 *
 * 浏览器原生 `EventSource` 只支持 GET，而我们的流式端点是 POST，
 * 因此手动读取 `ReadableStream` 并按 SSE 规范分帧解析。
 *
 * 与 `http.ts` 一致：自动附带 `Authorization` 头，401 时刷新令牌并重试一次。
 */

import { getAccessToken, notifyUnauthorized } from './authToken'

export interface RawSseEvent {
  event: string
  data: string
}

const FRAME_SEPARATOR = /\r?\n\r?\n/

/** 解析单个 SSE 帧（`event:` / `data:` 行；忽略注释与其它字段）。 */
function parseBlock(block: string): RawSseEvent | null {
  let event = 'message'
  const dataLines: string[] = []

  for (const rawLine of block.split(/\r?\n/)) {
    if (!rawLine || rawLine.startsWith(':')) continue // 空行 / 心跳注释
    const colon = rawLine.indexOf(':')
    const field = colon === -1 ? rawLine : rawLine.slice(0, colon)
    let value = colon === -1 ? '' : rawLine.slice(colon + 1)
    if (value.startsWith(' ')) value = value.slice(1)

    if (field === 'event') event = value
    else if (field === 'data') dataLines.push(value)
  }

  if (dataLines.length === 0) return null
  return { event, data: dataLines.join('\n') }
}

/**
 * 从缓冲区切出所有**完整**的 SSE 帧。
 * 返回已解析事件与尚未收全的尾部缓冲（需与后续 chunk 拼接）。
 */
export function parseSseBuffer(buffer: string): { events: RawSseEvent[]; rest: string } {
  const events: RawSseEvent[] = []
  let rest = buffer

  for (;;) {
    const match = FRAME_SEPARATOR.exec(rest)
    if (!match || match.index === undefined) break
    const block = rest.slice(0, match.index)
    rest = rest.slice(match.index + match[0].length)
    const parsed = parseBlock(block)
    if (parsed) events.push(parsed)
  }

  return { events, rest }
}

function safeJsonParse(text: string): unknown {
  try {
    return JSON.parse(text)
  } catch {
    return text
  }
}

export interface PostSseOptions {
  url: string
  body: unknown
  signal?: AbortSignal
  onEvent: (event: string, data: unknown) => void
}

function openStream(url: string, body: unknown, signal?: AbortSignal): Promise<Response> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'text/event-stream',
  }
  const token = getAccessToken()
  if (token) headers.Authorization = `Bearer ${token}`
  return fetch(url, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal,
    credentials: 'include', // 同源携带 refresh Cookie
  })
}

/** 发起 POST 请求并把 SSE 事件逐个回调给 `onEvent`。 */
export async function postSse({ url, body, signal, onEvent }: PostSseOptions): Promise<void> {
  let resp = await openStream(url, body, signal)
  if (resp.status === 401) {
    const refreshed = await notifyUnauthorized()
    if (refreshed) resp = await openStream(url, body, signal)
  }

  if (!resp.ok) {
    let message = `请求失败（HTTP ${resp.status}）`
    try {
      const payload: unknown = await resp.json()
      if (payload && typeof payload === 'object' && 'error' in payload) {
        const err = (payload as { error: unknown }).error
        if (typeof err === 'string') message = err
      }
    } catch {
      // 忽略非 JSON 错误体
    }
    throw new Error(message)
  }

  if (!resp.body) throw new Error('响应体为空，无法读取事件流')

  const reader = resp.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const { events, rest } = parseSseBuffer(buffer)
    buffer = rest
    for (const ev of events) onEvent(ev.event, safeJsonParse(ev.data))
  }

  buffer += decoder.decode()
  if (buffer.trim()) {
    const { events } = parseSseBuffer(`${buffer}\n\n`)
    for (const ev of events) onEvent(ev.event, safeJsonParse(ev.data))
  }
}
