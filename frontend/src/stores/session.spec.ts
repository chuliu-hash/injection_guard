import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { ChatMessage, ChatSession } from '@/types'

vi.mock('@/api/sessions', () => ({
  listSessions: vi.fn(),
  createSession: vi.fn(),
  getSession: vi.fn(),
  renameSession: vi.fn(),
  deleteSession: vi.fn(),
  clearSessions: vi.fn(),
  appendMessage: vi.fn(),
  patchMessage: vi.fn(),
  putDetection: vi.fn(),
}))

import * as api from '@/api/sessions'
import { useSessionStore } from '@/stores/session'

const mocked = vi.mocked(api)

function makeSession(id: string, title = '新会话'): ChatSession {
  return { id, title, createdAt: 1, updatedAt: 1, messages: [], detections: {} }
}

function makeMessage(
  id: string,
  role: 'user' | 'assistant',
  content: string,
  status: 'streaming' | 'done' = 'done',
): ChatMessage {
  return { id, role, content, status, createdAt: 1 }
}

describe('session store (server-backed)', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('loads sessions and selects the most recent', async () => {
    mocked.listSessions.mockResolvedValue([makeSession('a'), makeSession('b')])
    const store = useSessionStore()
    await store.loadSessions()
    expect(store.currentId).toBe('a')
    expect(store.sessions).toHaveLength(2)
  })

  it('creates a session on the server and selects it', async () => {
    mocked.createSession.mockResolvedValue(makeSession('s1'))
    const store = useSessionStore()
    const created = await store.createSession()
    expect(created.id).toBe('s1')
    expect(store.currentId).toBe('s1')
  })

  it('appends a user message and titles the session', async () => {
    mocked.createSession.mockResolvedValue(makeSession('s1'))
    mocked.appendMessage.mockResolvedValue(makeMessage('m1', 'user', '你好，请帮我写一段代码'))
    const store = useSessionStore()
    const session = await store.createSession()
    await store.appendMessage(session.id, 'user', '你好，请帮我写一段代码')
    expect(store.sessions[0].title.startsWith('你好')).toBe(true)
    expect(store.sessions[0].messages[0].id).toBe('m1')
  })

  it('keeps streamed deltas local and syncs only the final message', async () => {
    mocked.createSession.mockResolvedValue(makeSession('s1'))
    mocked.appendMessage.mockResolvedValue(makeMessage('m2', 'assistant', '', 'streaming'))
    mocked.patchMessage.mockResolvedValue(makeMessage('m2', 'assistant', '你好', 'done'))
    const store = useSessionStore()
    const session = await store.createSession()
    const assistant = await store.appendMessage(session.id, 'assistant', '')
    store.appendMessageContent(session.id, assistant.id, '你')
    store.appendMessageContent(session.id, assistant.id, '好')
    expect(store.sessions[0].messages[0].content).toBe('你好')
    expect(mocked.patchMessage).not.toHaveBeenCalled()
    await store.patchMessage(session.id, assistant.id, { status: 'done' })
    expect(mocked.patchMessage).toHaveBeenCalledWith('s1', 'm2', { status: 'done' })
  })

  it('attaches a detection record locally and on the server', async () => {
    mocked.createSession.mockResolvedValue(makeSession('s1'))
    mocked.putDetection.mockResolvedValue({ isSafe: true, stoppedAt: null })
    const store = useSessionStore()
    const session = await store.createSession()
    await store.attachDetection(session.id, 'm2', { isSafe: true, stoppedAt: null })
    expect(store.sessions[0].detections.m2.isSafe).toBe(true)
    expect(mocked.putDetection).toHaveBeenCalledWith('s1', 'm2', { isSafe: true, stoppedAt: null })
  })

  it('deletes a session and reselects another', async () => {
    mocked.createSession
      .mockResolvedValueOnce(makeSession('a'))
      .mockResolvedValueOnce(makeSession('b'))
    mocked.deleteSession.mockResolvedValue(undefined)
    // 删除当前会话后会重选并补拉新会话明细
    mocked.getSession.mockResolvedValue(makeSession('a'))
    const store = useSessionStore()
    const first = await store.createSession()
    const second = await store.createSession()
    await store.deleteSession(second.id)
    expect(store.sessions.map((s) => s.id)).toEqual([first.id])
    expect(store.currentId).toBe(first.id)
  })

  it('ensureSession reuses the current session without refetching', async () => {
    mocked.listSessions.mockResolvedValue([makeSession('s1')])
    mocked.createSession.mockResolvedValue(makeSession('s1'))
    const store = useSessionStore()
    const created = await store.createSession()
    const reused = await store.ensureSession()
    expect(reused.id).toBe(created.id)
    expect(mocked.createSession).toHaveBeenCalledTimes(1)
  })
})
