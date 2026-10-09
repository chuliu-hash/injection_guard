import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useSessionStore } from '@/stores/session'

describe('session store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('creates and selects a session', () => {
    const store = useSessionStore()
    const session = store.createSession()
    expect(store.currentId).toBe(session.id)
    expect(store.current?.id).toBe(session.id)
  })

  it('titles a session from the first user message', () => {
    const store = useSessionStore()
    const session = store.createSession()
    store.appendMessage(session.id, 'user', '你好，请帮我写一段代码')
    expect(store.sessions[0].title.startsWith('你好')).toBe(true)
  })

  it('appends streaming content and attaches a detection record', () => {
    const store = useSessionStore()
    const session = store.createSession()
    const assistant = store.appendMessage(session.id, 'assistant')
    expect(assistant.status).toBe('streaming')

    store.appendMessageContent(session.id, assistant.id, '你')
    store.appendMessageContent(session.id, assistant.id, '好')
    store.patchMessage(session.id, assistant.id, { status: 'done' })
    store.attachDetection(session.id, assistant.id, { isSafe: true, stoppedAt: null })

    const saved = store.sessions[0]
    expect(saved.messages[0].content).toBe('你好')
    expect(saved.messages[0].status).toBe('done')
    expect(saved.detections[assistant.id].isSafe).toBe(true)
  })

  it('deletes a session and reselects another', () => {
    const store = useSessionStore()
    const first = store.createSession()
    const second = store.createSession()
    store.deleteSession(second.id)
    expect(store.sessions.map((s) => s.id)).toEqual([first.id])
    expect(store.currentId).toBe(first.id)
  })

  it('ensureSession reuses the current session', () => {
    const store = useSessionStore()
    const created = store.createSession()
    expect(store.ensureSession().id).toBe(created.id)
    expect(store.sessions).toHaveLength(1)
  })
})
