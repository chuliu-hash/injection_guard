import type {
  ChatDefaults,
  ChatStreamRequest,
  DefendResponse,
  HealthInfo,
  TestServicesResponse,
} from '@/types'

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

/** 聊天模型的服务端默认值（不含密钥）；运行中修改 env 需重启后端。 */
export function fetchChatDefaults(signal?: AbortSignal): Promise<ChatDefaults> {
  return getJson<ChatDefaults>('/api/chat_defaults', signal)
}

/** 测试四个下游模型服务（输入/输出/RAG/聊天）的连通性。 */
export function testServices(
  options: {
    services?: string[]
    timeout?: number
    chat_api_url?: string
    chat_api_key?: string
    chat_model?: string
  } = {},
  signal?: AbortSignal,
): Promise<TestServicesResponse> {
  return postJson<TestServicesResponse>('/api/test_services', options, signal)
}
