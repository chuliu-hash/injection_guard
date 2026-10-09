"""防御 API 的请求 / 响应模型。"""

from __future__ import annotations

from typing import List, Optional

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

    ``stopped_at`` 仅在提前终止时非空（``input_detection`` / ``output_skipped``）；
    走完全流程时为 ``null``。``model_response`` 为本次实际使用的模型回复
    （请求提供或由聊天接口生成）。
    """

    is_safe: bool
    stopped_at: Optional[str] = None
    model_response: Optional[str] = None
    input_detection: Optional[InputDetectionResult] = None
    rag: Optional[RagResult] = None
    output_detection: Optional[OutputDetectionResult] = None
