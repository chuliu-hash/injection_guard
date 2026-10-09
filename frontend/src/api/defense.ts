import type { ChatStreamRequest, DefendResponse, HealthInfo } from '@/types'

import { apiUrl, getJson, postJson } from './http'
import { postSse } from './sse'

/** 流式对话：逐事件回调（input_detection / rag / delta / output_detection / done / error）。 */
export function chatStream(
  payload: ChatStreamRequest,
  onEvent: (event: string, data: unknown) => void,
  signal?: AbortSignal,
): Promise<void> {
  return postSse({ url: apiUrl('/api/chat_stream'), body: payload, signal, onEvent })
}

/** 非流式防御流水线（备用/脚本调用）。 */
export function promptDefend(payload: ChatStreamRequest, signal?: AbortSignal): Promise<DefendResponse> {
  return postJson<DefendResponse>('/api/prompt_defend', payload, signal)
}

/** 后端健康检查。 */
export function fetchHealth(signal?: AbortSignal): Promise<HealthInfo> {
  return getJson<HealthInfo>('/health', signal)
}
