/**
 * access token 的内存持有者（不放 localStorage，降低 XSS 窃取面）。
 *
 * `http.ts` 通过 `getAccessToken()` 取令牌；401 时调用 `notifyUnauthorized()`，
 * 由 auth store 注册的刷新处理器用 httpOnly Cookie 换新令牌。这样 api 层不反向依赖 store，
 * 避免循环引用。
 */

let accessToken: string | null = null
let refreshHandler: (() => Promise<boolean>) | null = null

export function getAccessToken(): string | null {
  return accessToken
}

export function setAccessToken(token: string | null): void {
  accessToken = token
}

export function registerRefreshHandler(handler: (() => Promise<boolean>) | null): void {
  refreshHandler = handler
}

/** 触发一次刷新；未注册或失败返回 false。 */
export async function notifyUnauthorized(): Promise<boolean> {
  if (!refreshHandler) return false
  try {
    return await refreshHandler()
  } catch {
    return false
  }
}
