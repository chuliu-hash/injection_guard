import { defineStore } from 'pinia'

import type { ChatMessage, ChatSession, DetectionRecord, MessageRole } from '@/types'
import { genId, truncate } from '@/utils/format'

interface SessionState {
  sessions: ChatSession[]
  currentId: string | null
}

const NEW_TITLE = '新会话'

export const useSessionStore = defineStore('session', {
  state: (): SessionState => ({
    sessions: [],
    currentId: null,
  }),

  getters: {
    current(state): ChatSession | undefined {
      return state.sessions.find((s) => s.id === state.currentId)
    },
    /** 按最近更新倒序（用于侧边栏）。 */
    sorted(state): ChatSession[] {
      return [...state.sessions].sort((a, b) => b.updatedAt - a.updatedAt)
    },
  },

  actions: {
    createSession(): ChatSession {
      const now = Date.now()
      const session: ChatSession = {
        id: genId(),
        title: NEW_TITLE,
        createdAt: now,
        updatedAt: now,
        messages: [],
        detections: {},
      }
      this.sessions.push(session)
      this.currentId = session.id
      return session
    },

    /** 返回当前会话；不存在则新建。 */
    ensureSession(): ChatSession {
      return this.current ?? this.createSession()
    },

    selectSession(id: string): void {
      if (this.sessions.some((s) => s.id === id)) this.currentId = id
    },

    renameSession(id: string, title: string): void {
      const session = this.sessions.find((s) => s.id === id)
      if (!session) return
      const next = title.trim()
      if (next) session.title = next
      session.updatedAt = Date.now()
    },

    deleteSession(id: string): void {
      const index = this.sessions.findIndex((s) => s.id === id)
      if (index === -1) return
      this.sessions.splice(index, 1)
      if (this.currentId === id) this.currentId = this.sessions[0]?.id ?? null
    },

    clearAll(): void {
      this.sessions = []
      this.currentId = null
    },

    appendMessage(sessionId: string, role: MessageRole, content = ''): ChatMessage {
      const session = this._require(sessionId)
      const message: ChatMessage = {
        id: genId(),
        role,
        content,
        createdAt: Date.now(),
        status: role === 'assistant' ? 'streaming' : 'done',
      }
      session.messages.push(message)
      session.updatedAt = Date.now()
      if (role === 'user' && session.title === NEW_TITLE) {
        session.title = truncate(content) || NEW_TITLE
      }
      return message
    },

    patchMessage(sessionId: string, messageId: string, patch: Partial<ChatMessage>): void {
      const session = this.sessions.find((s) => s.id === sessionId)
      const message = session?.messages.find((m) => m.id === messageId)
      if (!session || !message) return
      Object.assign(message, patch)
      session.updatedAt = Date.now()
    },

    appendMessageContent(sessionId: string, messageId: string, delta: string): void {
      const session = this.sessions.find((s) => s.id === sessionId)
      const message = session?.messages.find((m) => m.id === messageId)
      if (!session || !message) return
      message.content += delta
      session.updatedAt = Date.now()
    },

    attachDetection(sessionId: string, messageId: string, record: DetectionRecord): void {
      const session = this.sessions.find((s) => s.id === sessionId)
      if (!session) return
      session.detections = { ...session.detections, [messageId]: record }
      session.updatedAt = Date.now()
    },

    _require(sessionId: string): ChatSession {
      const session = this.sessions.find((s) => s.id === sessionId)
      if (!session) throw new Error(`会话不存在: ${sessionId}`)
      return session
    },
  },

  persist: true,
})
