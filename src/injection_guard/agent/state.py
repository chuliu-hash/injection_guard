"""LangGraph 防御流水线的状态定义。

``PipelineState`` 在整张状态图的各节点间流转：入口一次性写入请求参数，
下游节点只读；各阶段产物由对应节点写回，供后续节点与最终装配使用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, TypedDict

# 客户端断流后由编排层写入 ``stopped_at`` 的内部标记：仅用于让状态图就地结束，
# 不会出现在 SSE ``done`` 事件里（此时已无人消费）。
CANCELLED_STOP = "cancelled"


class PipelineState(TypedDict, total=False):
    """节点间共享的流水线状态（``total=False``：节点只返回自己产出的增量）。"""

    # ── 请求参数（入口写入，节点只读）────────────────────────────────
    user_input: str
    model_response: Optional[str]
    use_rag: bool
    sensitivity: Optional[str]
    use_bert: bool
    input_llm_api_url: Optional[str]
    input_llm_api_model: Optional[str]
    constitution_file: Optional[str]
    detector_api_key: Optional[str]
    detector_api_url: Optional[str]
    detector_model_name: Optional[str]
    temperature: Optional[float]
    max_tokens: Optional[int]
    skip_output_when_high_confidence: bool
    safe_skip_threshold: float
    # 流式增量输出检测间隔（token 估算值；0 = 关闭，只做生成结束后的整段检测）
    output_detect_interval: int
    # 流式模式：为 True 时节点通过 LangGraph 的 stream writer 推送 SSE 事件
    stream: bool

    # ── 各阶段产物 ───────────────────────────────────────────────────
    input_detection: Dict[str, Any]
    rag: Optional[Dict[str, Any]]
    final_prompt: str
    output_detection: Optional[Dict[str, Any]]
    is_safe: bool
    stopped_at: Optional[str]
