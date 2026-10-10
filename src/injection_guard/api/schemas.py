"""防御 API 的请求 / 响应模型。"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel


class DefendRequest(BaseModel):
    """``POST /api/prompt_defend`` 的请求体。

    ``user_input`` 必填；``model_response`` 可选（缺省时由流水线调用聊天接口生成）。
    其余字段全部 Optional：缺字段 / 空串等校验由 ``validate_defend_request`` 处理，
    以便返回与旧实现一致的中文错误消息（而非 FastAPI 默认的 422）。
    """

    user_input: Optional[str] = None
    model_response: Optional[str] = None

    use_bert: Optional[bool] = False

    # RAG 增强开关（None 时取全局 ENABLE_RAG）
    use_rag: Optional[bool] = None

    # 输入检测敏感度 low/medium/high（前端设置面板提供；None 时取全局 DEFAULT_SENSITIVITY）
    sensitivity: Optional[str] = None

    # 输出检测跳过策略（前端设置面板提供；None 时取 Settings 默认值）。
    # ⚠️ 客户端可控 = 可自行关闭输出侧检测，属安全面削弱。
    safe_skip_threshold: Optional[float] = None
    skip_output_when_high_confidence: Optional[bool] = None

    # 聊天生成参数（前端设置面板提供；None 时取 Settings 默认值）。
    # 仅当请求未带 model_response（即需要本服务生成回复）时生效。
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    # 输入检测器配置（仅首次初始化生效）
    input_llm_api_url: Optional[str] = None
    input_llm_api_model: Optional[str] = None

    # 输出检测器配置（仅首次初始化生效）
    constitution_file: Optional[str] = None
    detector_api_key: Optional[str] = None
    detector_api_url: Optional[str] = None
    detector_model_name: Optional[str] = None


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
    """RAG 增强结果。``enabled`` 为 True 表示本次请求实际执行了 RAG。"""

    enabled: bool
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


class TestServicesRequest(BaseModel):
    """``POST /api/test_services`` 的请求体（可省略整个 body）。

    校验（未知服务名 / 空列表 / timeout 越界）由路由层处理，以返回
    ``400 {"error": ...}``，与其它接口风格一致。
    """

    # 要测试的服务键：input / output / rag / chat；缺省全部
    services: Optional[List[str]] = None
    # 每个探测的超时（秒）；缺省 5.0，范围 0.5~60
    timeout: Optional[float] = None


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
