/** 认证接口（`/api/auth*`）。access token 由 store 存内存；refresh 走 httpOnly Cookie。 */
import type { AuthTokenResponse, LoginPayload, RegisterPayload, User } from '@/types'

import { request } from './http'

export function login(payload: LoginPayload): Promise<AuthTokenResponse> {
  return request<AuthTokenResponse>('/api/auth/login', {
    method: 'POST',
    body: payload,
    auth: false,
    retryOnUnauthorized: false,
  })
}

export function register(payload: RegisterPayload): Promise<AuthTokenResponse> {
  return request<AuthTokenResponse>('/api/auth/register', {
    method: 'POST',
    body: payload,
    auth: false,
    retryOnUnauthorized: false,
  })
}

/** 用 refresh Cookie 换新的 access token（同时轮换 refresh）。 */
export function refresh(): Promise<AuthTokenResponse> {
  return request<AuthTokenResponse>('/api/auth/refresh', {
    method: 'POST',
    body: {},
    auth: false,
    retryOnUnauthorized: false,
  })
}

export function logout(options: { allDevices?: boolean } = {}): Promise<void> {
  return request<void>('/api/auth/logout', {
    method: 'POST',
    body: { all_devices: Boolean(options.allDevices) },
    retryOnUnauthorized: false,
  })
}

export function fetchMe(): Promise<User> {
  return request<User>('/api/auth/me')
}
