"""防御 API 的请求 / 响应模型。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class DefendRequest(BaseModel):
    """``POST /api/prompt_defend`` 的请求体。

    ``user_input`` 必填；``model_response`` 可选（缺省时由流水线调用聊天接口生成）。
    其余字段全部 Optional：缺字段 / 空串等校验由 ``validate_defend_request`` 处理，
    以便返回与旧实现一致的中文错误消息（而非 FastAPI 默认的 422）。
    """

    user_input: Optional[str] = None
    model_response: Optional[str] = None

    # BERT 分类器组开关（None 时取全局 ENABLE_BERT）。
    # 模型是否加载由服务端 ENABLE_BERT 决定；请求体只能在服务端已启用时逐次关闭（false），
    # 要求 true 而服务端未启用会被忽略并记 warning。
    use_bert: Optional[bool] = None

    # RAG 增强开关（None 时取全局 ENABLE_RAG）
    use_rag: Optional[bool] = None

    # 输入检测敏感度 low/medium/high（前端设置面板提供；None 时取全局 DEFAULT_SENSITIVITY）
    sensitivity: Optional[str] = None

    # 输出检测跳过策略（前端设置面板提供；None 时取 Settings 默认值）。
    safe_skip_threshold: Optional[float] = None
    skip_output_when_high_confidence: Optional[bool] = None

    # 请求级聊天模型配置（前端设置面板提供；None 时取 Settings 默认值）。
    # API Key 只随单次请求传输，服务端不持久化、不写日志。
    chat_api_url: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_model: Optional[str] = None

    # 聊天生成参数（前端设置面板提供；None 时取 Settings 默认值）。
    # 仅当请求未带 model_response（即需要本服务生成回复）时生效。
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    seed: Optional[int] = None

    # 会话落库定位（可选）：流式生成结束时由**服务端**把助手消息正文写回会话，
    session_id: Optional[str] = None
    assistant_message_id: Optional[str] = None

    # LangGraph 检查点线程 id（可选）：客户端重试 / 恢复时复用同一 thread_id。
    # 不传则由服务端生成并随响应返回。
    thread_id: Optional[str] = None

    # 输入检测器配置（仅首次初始化生效）
    input_llm_api_url: Optional[str] = None
    input_llm_api_model: Optional[str] = None

    # 输出检测器配置（仅首次初始化生效）
    constitution_file: Optional[str] = None
    detector_api_key: Optional[str] = None
    detector_api_url: Optional[str] = None
    detector_model_name: Optional[str] = None

    def chat_options(self) -> Dict[str, Any]:
        """只返回请求实际提供的聊天配置，空白字符串按未提供处理。"""
        fields = (
            "chat_api_url",
            "chat_api_key",
            "chat_model",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "seed",
        )
        options: Dict[str, Any] = {}
        for field in fields:
            value = getattr(self, field)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            options[field.removeprefix("chat_")] = value.strip() if isinstance(value, str) else value
        return options


class ChatDefaultsResponse(BaseModel):
    """聊天模型的服务端默认值；**不返回 API Key**。"""

    api_url: str
    model: str
    api_key_configured: bool
    temperature: float
    max_tokens: int
    top_p: float
    frequency_penalty: float
    presence_penalty: float
    seed: Optional[int] = None


class InputDetectionResult(BaseModel):
    is_safe: bool
    risk_level: str
    category: str
    confidence: float
    p_safe: float
    p_attack: float
    decision_by: str
    # 本次实际生效的敏感度档位（请求未指定时即默认值）
    sensitivity: Optional[str] = None


class OutputDetectionResult(BaseModel):
    is_safe: bool
    failed_rules: List[str]
    total_rules: int


class RetrievedContext(BaseModel):
    """RAG 检索命中的一条安全示例（已剔除向量）。"""

    prompt: str
    response: str


class RagResult(BaseModel):
    """RAG 增强结果。

    ``enabled`` 为 True 表示本次请求**请求了** RAG；``degraded`` 为 True 表示请求了但检索失败
    已降级为原始输入（此时 ``reason="error"``，``error`` 是失败原因）。
    """

    enabled: bool
    degraded: bool = False
    reason: Optional[str] = None
    error: Optional[str] = None
    final_prompt: Optional[str] = None
    retrieved_contexts: List[RetrievedContext] = []


class DefendResponse(BaseModel):
    """防御流水线的完整响应。

    ``stopped_at`` 仅在提前终止时非空（``input_detection`` / ``output_detection`` /
    ``output_skipped``）；走完全流程且输出检测通过时为 ``null``。``model_response``
    为本次实际使用的模型回复（请求提供或由聊天接口生成），**不因输出不安全而被替换**。
    """

    is_safe: bool
    stopped_at: Optional[str] = None
    model_response: Optional[str] = None
    input_detection: Optional[InputDetectionResult] = None
    rag: Optional[RagResult] = None
    output_detection: Optional[OutputDetectionResult] = None
    # LangGraph 检查点线程与请求级追踪（P1）
    thread_id: Optional[str] = None
    trace: Optional[Dict[str, Any]] = None


class TestServicesRequest(BaseModel):
    """``POST /api/test_services`` 的请求体（可省略整个 body）。

    校验（未知服务名 / 空列表 / timeout 越界）由路由层处理，以返回
    ``400 {"error": ...}``，与其它接口风格一致。
    """

    # 要测试的服务键：input / output / rag / chat；缺省全部
    services: Optional[List[str]] = None
    # 每个探测的超时（秒）；缺省 60.0，范围 0.5~60
    timeout: Optional[float] = None
    # 可选：用前端当前聊天模型配置探测，而不是服务端默认值。
    chat_api_url: Optional[str] = None
    chat_api_key: Optional[str] = None
    chat_model: Optional[str] = None


class ServiceProbeResult(BaseModel):
    """单个下游服务的探测结果。``error`` 仅在 ``ok=False`` 时非空。"""

    ok: bool
    label: str
    url: str
    model: str
    latency_ms: Optional[float] = None
    error: Optional[str] = None


class TestServicesResponse(BaseModel):
    """连通性测试结果。``ok`` 为本次选中的所有服务是否全部通过。"""

    ok: bool
    timeout: float
    services: Dict[str, ServiceProbeResult]


# ── 会话持久化（/api/sessions*）───────────────────────────────────────
# 会话/消息/检测记录落库、按用户隔离；wire 格式沿用 snake_case，前端在 api 层转 camelCase。


class SessionSummary(BaseModel):
    """会话列表项（不含消息正文）。"""

    id: str
    title: str
    created_at: int
    updated_at: int
    message_count: int


class ChatMessageModel(BaseModel):
    id: str
    role: str
    content: str
    status: str
    error: Optional[str] = None
    created_at: int


class DetectionRecordModel(BaseModel):
    """助手消息对应的检测记录（三段结构原样透传）。"""

    is_safe: bool
    stopped_at: Optional[str] = None
    input_detection: Optional[Dict[str, Any]] = None
    rag: Optional[Dict[str, Any]] = None
    output_detection: Optional[Dict[str, Any]] = None


class SessionDetail(SessionSummary):
    messages: List[ChatMessageModel] = []
    # 以助手消息 id 为键的检测记录
    detections: Dict[str, DetectionRecordModel] = {}


class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]


class CreateSessionRequest(BaseModel):
    title: Optional[str] = None


class RenameSessionRequest(BaseModel):
    title: str


class CreateMessageRequest(BaseModel):
    role: str
    content: Optional[str] = None
    status: Optional[str] = None


class PatchMessageRequest(BaseModel):
    content: Optional[str] = None
    status: Optional[str] = None
    error: Optional[str] = None


class ClearSessionsResponse(BaseModel):
    deleted: int


# ── 认证（/api/auth*）────────────────────────────────────────────────


class UserModel(BaseModel):
    """对外的用户信息（**不含密码哈希**）。"""

    id: str
    username: str
    is_active: bool
    created_at: int


class RegisterRequest(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class RefreshRequest(BaseModel):
    """可选：非浏览器客户端可显式带 refresh_token；缺省则读 httpOnly Cookie。"""

    refresh_token: Optional[str] = None


class LogoutRequest(BaseModel):
    """``all_devices=True`` 时递增令牌版本，吊销该用户全部令牌。"""

    all_devices: bool = False


class TokenResponse(BaseModel):
    """登录 / 注册 / 刷新的统一响应。access_token 由前端内存持有；refresh 走 Cookie。"""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserModel
