/**
 * 会话持久化接口（`/api/sessions*`）。
 *
 * 会话/消息/检测记录保存在服务端数据库（按用户隔离），浏览器只做内存缓存。
 * 后端 wire 格式为 snake_case，本模块负责与前端 camelCase 的 `ChatSession` 互转。
 */
import type {
  ChatMessage,
  ChatSession,
  DetectionRecord,
  MessageRole,
  MessageStatus,
  StoppedAt,
} from '@/types'

import { deleteJson, getJson, patchJson, postJson, putJson } from './http'

interface SessionSummaryWire {
  id: string
  title: string
  created_at: number
  updated_at: number
  message_count: number
}

interface ChatMessageWire {
  id: string
  role: MessageRole
  content: string
  status: MessageStatus
  error?: string | null
  created_at: number
}

interface DetectionRecordWire {
  is_safe: boolean
  stopped_at: StoppedAt
  input_detection?: DetectionRecord['inputDetection'] | null
  rag?: DetectionRecord['rag'] | null
  output_detection?: DetectionRecord['outputDetection'] | null
}

interface SessionDetailWire extends SessionSummaryWire {
  messages: ChatMessageWire[]
  detections: Record<string, DetectionRecordWire>
}

function toMessage(wire: ChatMessageWire): ChatMessage {
  return {
    id: wire.id,
    role: wire.role,
    content: wire.content,
    status: wire.status,
    createdAt: wire.created_at,
    ...(wire.error ? { error: wire.error } : {}),
  }
}

function toDetection(wire: DetectionRecordWire): DetectionRecord {
  return {
    isSafe: wire.is_safe,
    stoppedAt: wire.stopped_at ?? null,
    inputDetection: wire.input_detection ?? undefined,
    rag: wire.rag ?? undefined,
    outputDetection: wire.output_detection ?? undefined,
  }
}

/** 列表项 -> 领域对象（不含消息正文，messages/detections 为空占位）。 */
function toSessionSummary(wire: SessionSummaryWire): ChatSession {
  return {
    id: wire.id,
    title: wire.title,
    createdAt: wire.created_at,
    updatedAt: wire.updated_at,
    messages: [],
    detections: {},
  }
}

function toSessionDetail(wire: SessionDetailWire): ChatSession {
  return {
    ...toSessionSummary(wire),
    messages: wire.messages.map(toMessage),
    detections: Object.fromEntries(
      Object.entries(wire.detections).map(([id, record]) => [id, toDetection(record)]),
    ),
  }
}

export async function listSessions(signal?: AbortSignal): Promise<ChatSession[]> {
  const data = await getJson<{ sessions: SessionSummaryWire[] }>('/api/sessions', signal)
  return data.sessions.map(toSessionSummary)
}

export async function createSession(title?: string, signal?: AbortSignal): Promise<ChatSession> {
  const wire = await postJson<SessionDetailWire>('/api/sessions', title ? { title } : {}, signal)
  return toSessionDetail(wire)
}

export function getSession(id: string, signal?: AbortSignal): Promise<ChatSession> {
  return getJson<SessionDetailWire>(`/api/sessions/${id}`, signal).then(toSessionDetail)
}

export async function renameSession(id: string, title: string): Promise<ChatSession> {
  const wire = await patchJson<SessionSummaryWire>(`/api/sessions/${id}`, { title })
  return toSessionSummary(wire)
}

export function deleteSession(id: string): Promise<void> {
  return deleteJson(`/api/sessions/${id}`)
}

export async function clearSessions(): Promise<number> {
  const data = await deleteJson<{ deleted: number }>('/api/sessions')
  return data.deleted
}

export function appendMessage(
  sessionId: string,
  role: MessageRole,
  content = '',
  status?: MessageStatus,
): Promise<ChatMessage> {
  return postJson<ChatMessageWire>(`/api/sessions/${sessionId}/messages`, {
    role,
    content,
    status,
  }).then(toMessage)
}

export function patchMessage(
  sessionId: string,
  messageId: string,
  patch: Partial<Pick<ChatMessage, 'content' | 'status' | 'error'>>,
): Promise<ChatMessage> {
  return patchJson<ChatMessageWire>(`/api/sessions/${sessionId}/messages/${messageId}`, patch).then(
    toMessage,
  )
}

export function putDetection(
  sessionId: string,
  messageId: string,
  record: DetectionRecord,
): Promise<DetectionRecord> {
  return putJson<DetectionRecordWire>(
    `/api/sessions/${sessionId}/messages/${messageId}/detection`,
    {
      is_safe: record.isSafe,
      stopped_at: record.stoppedAt,
      input_detection: record.inputDetection ?? null,
      rag: record.rag ?? null,
      output_detection: record.outputDetection ?? null,
    },
  ).then(toDetection)
}
