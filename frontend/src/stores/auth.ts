import { defineStore } from 'pinia'

import * as authApi from '@/api/auth'
import { setAccessToken } from '@/api/authToken'
import { useSessionStore } from '@/stores/session'
import type { AuthTokenResponse, User } from '@/types'

interface AuthState {
  user: User | null
  /** access token 仅存内存（刷新页面后靠 refresh Cookie 恢复）。 */
  accessToken: string | null
  /** `ensureReady` 是否已执行过（避免每次导航都尝试恢复）。 */
  ready: boolean
  loading: boolean
}

export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({ user: null, accessToken: null, ready: false, loading: false }),

  getters: {
    isAuthenticated: (state): boolean => Boolean(state.accessToken),
  },

  actions: {
    _apply(payload: AuthTokenResponse): void {
      this.accessToken = payload.access_token
      this.user = payload.user
      setAccessToken(payload.access_token)
    },

    /** 清空本地登录态与业务缓存（不发请求）。 */
    _clearLocal(): void {
      this.accessToken = null
      this.user = null
      setAccessToken(null)
      useSessionStore().reset()
    },

    async login(username: string, password: string): Promise<void> {
      this.loading = true
      try {
        this._apply(await authApi.login({ username, password }))
      } finally {
        this.loading = false
      }
    },

    async register(username: string, password: string): Promise<void> {
      this.loading = true
      try {
        this._apply(await authApi.register({ username, password }))
      } finally {
        this.loading = false
      }
    },

    /** 用 refresh Cookie 恢复会话；失败则清空本地状态。 */
    async restore(): Promise<boolean> {
      try {
        this._apply(await authApi.refresh())
        return true
      } catch {
        this._clearLocal()
        return false
      }
    },

    /** 应用启动/路由守卫调用：只尝试恢复一次，返回是否已登录。 */
    async ensureReady(): Promise<boolean> {
      if (this.ready) return this.isAuthenticated
      this.ready = true
      return await this.restore()
    },

    /** http 层 401 时调用：刷新成功返回 true（供重试）。 */
    async refreshSession(): Promise<boolean> {
      try {
        this._apply(await authApi.refresh())
        return true
      } catch {
        this._clearLocal()
        return false
      }
    },

    async logout(allDevices = false): Promise<void> {
      try {
        await authApi.logout({ allDevices })
      } catch {
        // 本地登出仍要生效（例如令牌已过期）
      } finally {
        this._clearLocal()
        this.ready = true
      }
    },
  },
})
