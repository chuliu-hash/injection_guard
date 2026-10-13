import { defineStore } from 'pinia'

import * as sessionsApi from '@/api/sessions'
import type { ChatMessage, ChatSession, DetectionRecord, MessageRole } from '@/types'
import { truncate } from '@/utils/format'

interface SessionState {
  /** 会话列表（明细懒加载：仅当前会话含完整 messages/detections）。 */
  sessions: ChatSession[]
  currentId: string | null
  /** 是否已从服务端拉取过会话列表。 */
  loaded: boolean
  loading: boolean
  /** 已加载完整明细的会话 id（避免重复请求）。 */
  loadedIds: string[]
}

const NEW_TITLE = '新会话'

/**
 * 进行中的列表请求（模块级，不放进 state：它是「进程内共享的 Promise」，不是数据）。
 *
 * 侧栏（SessionList）与聊天页（ChatView）在同一次导航里**同时挂载**、各自调用
 * ``ensureLoaded()``：若第二次调用看到 ``loading`` 就直接返回，``activate()`` 会误判成
 * 「没有任何会话」而新建一个空会话，用户看到的就是「历史记录不见了、聊天区一片空白」。
 * 因此并发调用共享同一次请求。
 */
let listInflight: Promise<void> | null = null

/** 登录态代次：``reset()``（登出 / 切换账号）后，先前发出的列表结果必须丢弃。 */
let listGeneration = 0

/**
 * 会话状态：**服务端为权威数据源**（`/api/sessions`，按用户隔离）。
 * 本 store 只做内存缓存与乐观更新；流式增量（appendMessageContent）仅改本地，
 * 结束时再用 patchMessage 落库，避免逐 token 打服务端。
 */
export const useSessionStore = defineStore('session', {
  state: (): SessionState => ({
    sessions: [],
    currentId: null,
    loaded: false,
    loading: false,
    loadedIds: [],
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
    /** 拉取会话列表；并发调用共享同一次请求（见模块级 ``listInflight``）。 */
    async loadSessions(): Promise<void> {
      if (listInflight) return listInflight
      this.loading = true
      const generation = listGeneration
      listInflight = this._fetchSessions(generation)
      return listInflight
    },

    /** 实际拉取；当前选中项失效时回落到最近一个。 */
    async _fetchSessions(generation: number): Promise<void> {
      try {
        const list = await sessionsApi.listSessions()
        if (generation !== listGeneration) return // 期间已登出 / 切换账号：丢弃过期结果
        this.sessions = list
        this.loaded = true
        if (!this.currentId || !list.some((s) => s.id === this.currentId)) {
          this.currentId = list[0]?.id ?? null
        }
      } finally {
        this.loading = false
        listInflight = null
      }
    },

    async ensureLoaded(): Promise<void> {
      if (!this.loaded) await this.loadSessions()
    },

    /** 打开指定会话（缺省为最近一个），无会话则新建。返回被打开的会话。 */
    async activate(sessionId?: string): Promise<ChatSession> {
      await this.ensureLoaded()
      const target =
        sessionId && this.sessions.some((s) => s.id === sessionId)
          ? sessionId
          : (this.sessions[0]?.id ?? null)
      if (target) {
        await this.selectSession(target)
        return this.current as ChatSession
      }
      return this.createSession()
    },

    /** 兼容旧调用点：返回当前会话，没有则新建。 */
    ensureSession(): Promise<ChatSession> {
      return this.activate(this.currentId ?? undefined)
    },

    async selectSession(id: string): Promise<void> {
      if (!this.sessions.some((s) => s.id === id)) return
      const alreadyLoaded = this.currentId === id && this.loadedIds.includes(id)
      this.currentId = id
      if (!alreadyLoaded) await this.loadDetail(id)
    },

    async loadDetail(id: string): Promise<void> {
      const detail = await sessionsApi.getSession(id)
      const index = this.sessions.findIndex((s) => s.id === id)
      if (index === -1) this.sessions.push(detail)
      else this.sessions[index] = detail
      if (!this.loadedIds.includes(id)) this.loadedIds.push(id)
    },

    async createSession(): Promise<ChatSession> {
      const session = await sessionsApi.createSession()
      this.sessions.unshift(session)
      this.loadedIds.push(session.id)
      this.currentId = session.id
      return session
    },

    async renameSession(id: string, title: string): Promise<void> {
      const next = title.trim()
      if (!next) return
      const updated = await sessionsApi.renameSession(id, next)
      const session = this.sessions.find((s) => s.id === id)
      if (session) {
        session.title = updated.title
        session.updatedAt = updated.updatedAt
      }
    },

    async deleteSession(id: string): Promise<void> {
      await sessionsApi.deleteSession(id)
      const index = this.sessions.findIndex((s) => s.id === id)
      if (index === -1) return
      this.sessions.splice(index, 1)
      this.loadedIds = this.loadedIds.filter((item) => item !== id)
      if (this.currentId === id) {
        this.currentId = this.sessions[0]?.id ?? null
        // 新选中的会话明细尚未加载，补齐以免聊天区空白
        if (this.currentId) await this.loadDetail(this.currentId)
      }
    },

    async clearAll(): Promise<void> {
      await sessionsApi.clearSessions()
      this.reset()
    },

    /** 仅清本地缓存（登出时调用，不发服务端请求）。 */
    reset(): void {
      this.sessions = []
      this.currentId = null
      this.loaded = false
      this.loading = false
      this.loadedIds = []
      // 让仍在途的列表请求作废，避免把上一个账号的会话写回缓存
      listGeneration += 1
      listInflight = null
    },

    async appendMessage(sessionId: string, role: MessageRole, content = ''): Promise<ChatMessage> {
      const message = await sessionsApi.appendMessage(sessionId, role, content)
      const session = this._require(sessionId)
      session.messages.push(message)
      session.updatedAt = message.createdAt
      if (role === 'user' && session.title === NEW_TITLE) {
        session.title = truncate(content) || NEW_TITLE
      }
      return message
    },

    /** 流式增量：仅改本地缓存（不打服务端，结束时统一 patch）。 */
    appendMessageContent(sessionId: string, messageId: string, delta: string): void {
      const message = this._findMessage(sessionId, messageId)
      if (!message) return
      message.content += delta
    },

    /** 落库消息变更（内容 / 状态 / 错误）；失败不阻断 UI，本地状态保留待下次加载对齐。 */
    async patchMessage(
      sessionId: string,
      messageId: string,
      patch: Partial<Pick<ChatMessage, 'content' | 'status' | 'error'>>,
    ): Promise<void> {
      const message = this._findMessage(sessionId, messageId)
      if (message) Object.assign(message, patch)
      await this._sync(() => sessionsApi.patchMessage(sessionId, messageId, patch))
    },

    async attachDetection(
      sessionId: string,
      messageId: string,
      record: DetectionRecord,
    ): Promise<void> {
      const session = this.sessions.find((s) => s.id === sessionId)
      if (!session) return
      session.detections = { ...session.detections, [messageId]: record }
      await this._sync(() => sessionsApi.putDetection(sessionId, messageId, record))
    },

    async _sync(operation: () => Promise<unknown>): Promise<void> {
      try {
        await operation()
      } catch (error) {
        // 服务端同步失败不阻断聊天；本地缓存仍可见，刷新后以服务端为准。
        console.warn('[session] 同步到服务端失败:', error)
      }
    },

    _findMessage(sessionId: string, messageId: string): ChatMessage | undefined {
      return this.sessions.find((s) => s.id === sessionId)?.messages.find((m) => m.id === messageId)
    },

    _require(sessionId: string): ChatSession {
      const session = this.sessions.find((s) => s.id === sessionId)
      if (!session) throw new Error(`会话不存在: ${sessionId}`)
      return session
    },
  },
})
