import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { AuthTokenResponse } from '@/types'

vi.mock('@/api/auth', () => ({
  login: vi.fn(),
  register: vi.fn(),
  refresh: vi.fn(),
  logout: vi.fn(),
  fetchMe: vi.fn(),
}))

import * as authApi from '@/api/auth'
import { getAccessToken } from '@/api/authToken'
import { useAuthStore } from '@/stores/auth'

const mocked = vi.mocked(authApi)

function tokenResponse(username = 'alice'): AuthTokenResponse {
  return {
    access_token: 'tok-1',
    token_type: 'bearer',
    expires_in: 1800,
    user: { id: 'u1', username, is_active: true, created_at: 1 },
  }
}

describe('auth store', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('logs in and exposes the token to the http layer', async () => {
    mocked.login.mockResolvedValue(tokenResponse())
    const store = useAuthStore()
    await store.login('alice', 'secret')
    expect(store.isAuthenticated).toBe(true)
    expect(store.user?.username).toBe('alice')
    expect(getAccessToken()).toBe('tok-1')
  })

  it('restores the session from the refresh cookie', async () => {
    mocked.refresh.mockResolvedValue(tokenResponse())
    const store = useAuthStore()
    expect(await store.ensureReady()).toBe(true)
    expect(store.isAuthenticated).toBe(true)
  })

  it('clears state when restore fails', async () => {
    mocked.refresh.mockRejectedValue(new Error('401'))
    const store = useAuthStore()
    expect(await store.ensureReady()).toBe(false)
    expect(store.isAuthenticated).toBe(false)
    expect(getAccessToken()).toBeNull()
  })

  it('only attempts a restore once', async () => {
    mocked.refresh.mockResolvedValue(tokenResponse())
    const store = useAuthStore()
    await store.ensureReady()
    await store.ensureReady()
    expect(mocked.refresh).toHaveBeenCalledTimes(1)
  })

  it('refreshes on demand for 401 retries', async () => {
    mocked.refresh.mockResolvedValue(tokenResponse())
    const store = useAuthStore()
    expect(await store.refreshSession()).toBe(true)
    expect(getAccessToken()).toBe('tok-1')
  })

  it('logs out locally even if the API call fails', async () => {
    mocked.login.mockResolvedValue(tokenResponse())
    mocked.logout.mockRejectedValue(new Error('expired'))
    const store = useAuthStore()
    await store.login('alice', 'secret')
    await store.logout()
    expect(store.isAuthenticated).toBe(false)
    expect(getAccessToken()).toBeNull()
  })
})
