/** 与后端 `api/schemas.py` 及 SSE 事件契约对应的类型定义。 */

/** 输入检测敏感度档位（对应后端 `config.SENSITIVITY_LEVELS`）。 */
export type SensitivityLevel = 'low' | 'medium' | 'high'

export interface InputDetection {
  is_safe: boolean
  risk_level: string
  category: string
  confidence: number
  p_safe: number
  p_attack: number
  decision_by: string
  /** 本次实际生效的敏感度档位（请求未指定时即后端默认值）。 */
  sensitivity?: SensitivityLevel
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

export type StoppedAt = 'input_detection' | 'output_detection' | 'output_skipped' | null

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
  /** 输入检测敏感度（设置面板提供；缺省由后端的 DEFAULT_SENSITIVITY 决定）。 */
  sensitivity?: SensitivityLevel
  model_response?: string
  /** 聊天生成参数（设置面板提供；缺省由后端的 Settings 默认值决定）。 */
  temperature?: number
  max_tokens?: number
  /** 输出检测跳过策略（设置面板提供；缺省由后端 Settings 默认值决定）。 */
  safe_skip_threshold?: number
  skip_output_when_high_confidence?: boolean
}

export interface HealthInfo {
  status: string
  default_sensitivity: SensitivityLevel
  input_max_concurrency: number
  output_max_concurrency: number
  chat_max_concurrency: number
  enable_rag: boolean
  [key: string]: unknown
}

/** 可测试的下游模型服务键（与后端 connectivity.SERVICE_KEYS 一致）。 */
export type ServiceKey = 'input' | 'output' | 'rag' | 'chat'

export interface ServiceProbeResult {
  ok: boolean
  label: string
  url: string
  model: string
  latency_ms?: number | null
  /** 仅 ok=false 时非空。 */
  error?: string | null
}

export interface TestServicesResponse {
  /** 本次选中的所有服务是否全部通过。 */
  ok: boolean
  timeout: number
  services: Partial<Record<ServiceKey, ServiceProbeResult>>
}

// ── 认证 ──────────────────────────────────────────────────────────────

/** 当前登录用户（对应后端 `api/schemas.py::UserModel`，不含密码）。 */
export interface User {
  id: string
  username: string
  is_active: boolean
  created_at: number
}

/** 登录 / 注册 / 刷新的统一响应；refresh token 走 httpOnly Cookie，不出现在此处。 */
export interface AuthTokenResponse {
  access_token: string
  token_type: string
  expires_in: number
  user: User
}

export interface LoginPayload {
  username: string
  password: string
}

export type RegisterPayload = LoginPayload
