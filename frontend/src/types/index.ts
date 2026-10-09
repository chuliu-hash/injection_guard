/** 与后端 `api/schemas.py` 及 SSE 事件契约对应的类型定义。 */

export interface InputDetection {
  is_safe: boolean
  risk_level: string
  category: string
  confidence: number
  p_safe: number
  p_attack: number
  decision_by: string
}

export interface OutputDetection {
  is_safe: boolean
  failed_rules: string[]
  total_rules: number
}

export interface RetrievedContext {
  prompt: string
  response: string
}

export interface RagResult {
  enabled: boolean
  final_prompt?: string | null
  retrieved_contexts: RetrievedContext[]
}

export interface DefendResponse {
  is_safe: boolean
  stopped_at: StoppedAt
  model_response: string | null
  input_detection: InputDetection | null
  rag: RagResult | null
  output_detection: OutputDetection | null
}

export type StoppedAt = 'input_detection' | 'output_skipped' | null

/** SSE `done` 事件的负载。 */
export interface DonePayload {
  is_safe: boolean
  stopped_at: StoppedAt
  model_response: string | null
}

export type SseEventName =
  | 'input_detection'
  | 'rag'
  | 'delta'
  | 'output_detection'
  | 'done'
  | 'error'

/** 一条消息对应的检测结果（前端聚合后持久化）。 */
export interface DetectionRecord {
  isSafe: boolean
  stoppedAt: StoppedAt
  inputDetection?: InputDetection
  rag?: RagResult
  outputDetection?: OutputDetection
}

export type MessageRole = 'user' | 'assistant'
export type MessageStatus = 'streaming' | 'done' | 'error'

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  createdAt: number
  status: MessageStatus
  error?: string
}

export interface ChatSession {
  id: string
  title: string
  createdAt: number
  updatedAt: number
  messages: ChatMessage[]
  /** 以 assistant 消息 id 为键的检测记录。 */
  detections: Record<string, DetectionRecord>
}

export interface ChatStreamRequest {
  user_input: string
  use_rag?: boolean
  model_response?: string
}

export interface HealthInfo {
  status: string
  fixed_sensitivity: string
  input_max_concurrency: number
  output_max_concurrency: number
  chat_max_concurrency: number
  enable_rag: boolean
  [key: string]: unknown
}
