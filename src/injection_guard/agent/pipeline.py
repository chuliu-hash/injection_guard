"""基于 LangGraph 的防御流水线编排（企业级 Agent 工作流）。

状态图结构（条件边实现提前拦截 / 跳过）::

    START → input_detection ─┬─(不安全)→ END
                             └─(安全)→ rag_retrieval → generation → skip_policy
                                                    ├─(高置信)→ END
                                                    └──────→ output_detection → END

各节点只做「编排」：真正的检测 / 检索 / 生成仍由
``DefenseService`` 的并发受控方法（``_run_*_guarded``）执行，因此并发闸门、
RAG 降级、异常分类（``ServiceBusyError`` / ``DetectionError``）语义与旧实现一致。

检测节点挂了 ``RetryPolicy``（见 :func:`_retry_policy`）：只重试 ``DetectionError``，
``ServiceBusyError`` 保持「快速失败 -> 503」不重试。生成节点不重试（重放会重复生成）。

- 流式生成期间，每累计约 ``output_detect_interval`` 个 token 就对**已生成内容**做一次
  输出检测（估算见 ``llm_utils.estimate_tokens``），命中不安全立刻中断生成并推送
  ``output_detection`` 事件；生成结束后仍会做一次整段检测（覆盖不足一个间隔的尾巴）。
- 同步模式（``stream=False``）：``DefenseAgent.run`` 用 ``graph.invoke`` 一次跑完；
- 流式模式（``stream=True``）：``DefenseAgent.stream`` 用
  ``graph.stream(stream_mode=["custom", "values"])`` 逐节点推送事件，节点内部经
  ``get_stream_writer()`` 发送 SSE 事件。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, List, Optional, Protocol, Tuple

from langchain_core.runnables import RunnableConfig

from ..errors import DetectionError, ServiceBusyError
from ..llm_utils import estimate_tokens
from .state import CANCELLED_STOP, PipelineState

logger = logging.getLogger(__name__)


class PipelineSteps(Protocol):
    """流水线各阶段的实际执行者（由 ``DefenseService`` 实现）。"""

    def _run_input_detection_guarded(self, **kwargs: Any) -> Dict[str, Any]: ...

    def _run_output_detection_guarded(self, **kwargs: Any) -> Any: ...

    def _run_rag(self, user_input: str) -> Dict[str, Any]: ...

    def _run_chat_guarded(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str: ...

    def _run_chat_stream_guarded(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Iterator[str]: ...


def _emit(state: PipelineState, event: str, data: Any) -> None:
    """流式模式下通过 LangGraph 的自定义流推送一个 SSE 事件。"""
    if not state.get("stream"):
        return
    from langgraph.config import get_stream_writer

    try:
        writer = get_stream_writer()
    except Exception:  # noqa: BLE001 - 不在状态图运行时（例如直接调用节点做单测）没有事件通道
        logger.debug("[流式] 无 stream writer，跳过事件 %s", event)
        return
    writer({"event": event, "data": data})


def _cancel_event(config: Optional[Dict[str, Any]]):
    """取出编排层注入的取消信号（客户端断流置位）。"""
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("cancel_event")


def _detection_payload(result: Any) -> Dict[str, Any]:
    """把输出检测结果整理成状态 / SSE 事件里统一的字典。"""
    return {
        "is_safe": result.is_safe,
        "failed_rules": result.failed_rules,
        "total_rules": result.total_rules,
    }


def _skips_output_detection(state: PipelineState) -> bool:
    """输入高置信是否跳过输出检测（收尾策略与流式增量检测共用同一判断）。

    两者必须一致：既然按策略跳过了收尾检测，就不该在流式过程中额外发起增量检测。
    """
    if not state.get("skip_output_when_high_confidence"):
        return False
    input_detection = state.get("input_detection") or {}
    return input_detection.get("p_safe", 0.0) >= state.get("safe_skip_threshold", 1.0)


# ── 节点 ────────────────────────────────────────────────────────────


def input_detection_node(
    state: PipelineState, steps: PipelineSteps, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 1：输入检测（命中即拦截，不再进入后续阶段）。"""
    try:
        detection = steps._run_input_detection_guarded(
            text=state["user_input"],
            use_bert=state.get("use_bert", False),
            llm_api_url=state.get("input_llm_api_url"),
            llm_api_model=state.get("input_llm_api_model"),
            sensitivity=state.get("sensitivity"),
        )
    except ServiceBusyError:
        logger.warning("[503] input_detection busy")
        raise
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"输入检测失败: {exc}") from exc

    # 输入检测链路内部异常：不放行
    if detection.get("detection_error"):
        raise DetectionError(f"输入检测异常: {detection['detection_error']}")

    _emit(state, "input_detection", detection)

    if not detection.get("is_safe"):
        return {
            "input_detection": detection,
            "is_safe": False,
            "stopped_at": "input_detection",
        }
    return {"input_detection": detection}


def rag_retrieval_node(
    state: PipelineState, steps: PipelineSteps, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 2：可选的 RAG 安全上下文增强（失败降级为原始输入）。

    ``rag`` 事件区分四种情况，便于前端与排障：

    - 未启用：``{"enabled": False, "reason": "disabled"}``；
    - 已启用且成功：``{"enabled": True, "degraded": False, ...}``；
    - 已启用但没命中安全示例：``{"enabled": True, "degraded": False,
      "reason": "no_match", "final_prompt": None}``（不拼接上下文，直接用原始输入）；
    - 已启用但降级：``{"enabled": True, "degraded": True, "reason": "error", "error": ...}``
      （检索异常，已回退为原始输入，不阻断主流程）。
    """
    rag_info: Optional[Dict[str, Any]] = None
    final_prompt = state["user_input"]

    if not state.get("use_rag"):
        event: Dict[str, Any] = {
            "enabled": False,
            "degraded": False,
            "reason": "disabled",
            "final_prompt": None,
            "retrieved_contexts": [],
        }
    else:
        try:
            rag_info = steps._run_rag(state["user_input"])
            final_prompt = rag_info["final_prompt"]
            # 保留检索层给出的 reason（如 no_match）；只有真正命中时它才是 None
            event = {**rag_info, "degraded": False, "reason": rag_info.get("reason")}
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RAG] 增强失败，降级为原始输入: %s", exc)
            # 降级也写回状态：非流式响应 / 检测记录里能看出「开了但没生效」而不是「没开」
            rag_info = {
                "enabled": True,
                "degraded": True,
                "reason": "error",
                "error": str(exc)[:200],
                "final_prompt": None,
                "retrieved_contexts": [],
            }
            event = rag_info
            final_prompt = state["user_input"]

    _emit(state, "rag", event)
    return {"rag": rag_info, "final_prompt": final_prompt}


def generation_node(
    state: PipelineState, steps: PipelineSteps, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 3：生成回复（请求已带 ``model_response`` 时跳过聊天调用）。"""
    if state.get("model_response") is not None:
        return {}

    prompt = state.get("final_prompt") or state["user_input"]
    max_tokens = state.get("max_tokens")
    temperature = state.get("temperature")
    # 多轮记忆：历史只影响生成，不参与输入检测（信任边界见 README「多轮记忆」）
    history = state.get("history") or None

    if not state.get("stream"):
        try:
            return {
                "model_response": steps._run_chat_guarded(
                    prompt, max_tokens=max_tokens, temperature=temperature, history=history
                )
            }
        except ServiceBusyError:
            logger.warning("[503] chat busy")
            raise
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"聊天生成失败: {exc}") from exc

    cancel_event = _cancel_event(config)
    interval = int(state.get("output_detect_interval") or 0)
    incremental = interval > 0 and not _skips_output_detection(state)
    text = ""
    tokens_since_check = 0
    checked_tokens = 0
    upstream = steps._run_chat_stream_guarded(
        prompt, max_tokens=max_tokens, temperature=temperature, history=history
    )
    try:
        for delta in upstream:
            if cancel_event is not None and cancel_event.is_set():
                # 客户端已断流：就地结束状态图，避免继续输出检测
                logger.info("[流式] 客户端断开，提前终止聊天生成")
                return {
                    "model_response": text,
                    "is_safe": False,
                    "stopped_at": CANCELLED_STOP,
                }
            text += delta
            _emit(state, "delta", {"content": delta})

            if not incremental:
                continue

            # ── 增量输出检测：每累计约 interval 个 token 检查一次已生成内容 ──
            tokens_since_check += estimate_tokens(delta)
            if tokens_since_check < interval:
                continue
            # 扣掉一个间隔、保留余数，避免每次检测都丢弃不足一个间隔的 token
            tokens_since_check -= interval
            checked_tokens += interval
            try:
                result = steps._run_output_detection_guarded(
                    text=state["user_input"],
                    answer=text,
                    constitution_file=state.get("constitution_file"),
                    api_key=state.get("detector_api_key"),
                    base_url=state.get("detector_api_url"),
                    model_name=state.get("detector_model_name"),
                )
            except ServiceBusyError:
                logger.warning("[503] output_detection busy（增量检测）")
                raise
            except Exception as exc:  # noqa: BLE001 - 与收尾检测一致：检测失败默认放行
                logger.warning("[流式] 增量输出检测失败（继续生成）: %s", exc)
                continue

            if not result.is_safe:
                detection = _detection_payload(result)
                _emit(state, "output_detection", detection)
                logger.warning(
                    "[输出拦截] 流式增量检测命中，提前中断生成（约 %d token 处，命中类别: %s）",
                    checked_tokens,
                    ", ".join(result.failed_rules) or "未给出",
                )
                return {
                    "model_response": text,
                    "output_detection": detection,
                    "is_safe": False,
                    "stopped_at": "output_detection",
                }
    except ServiceBusyError:
        logger.warning("[503] chat busy")
        raise
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"聊天生成失败: {exc}") from exc
    finally:
        # 提前终止时同步关闭上游 HTTP 流，尽快释放并发名额
        close = getattr(upstream, "close", None)
        if close is not None:
            close()

    return {"model_response": text}


def skip_policy_node(
    state: PipelineState, steps: Optional[PipelineSteps] = None, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 4：输入高置信时跳过输出检测（仍保留生成的回复）。"""
    if state.get("stopped_at"):
        # 上游已终止（如流式增量检测命中）：不要覆盖终止原因
        return {}
    if _skips_output_detection(state):
        return {"is_safe": True, "stopped_at": "output_skipped"}
    return {}


def output_detection_node(
    state: PipelineState, steps: PipelineSteps, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 5：输出检测（"User Input" 用原始 user_input）。"""
    try:
        result = steps._run_output_detection_guarded(
            text=state["user_input"],
            answer=state["model_response"],
            constitution_file=state.get("constitution_file"),
            api_key=state.get("detector_api_key"),
            base_url=state.get("detector_api_url"),
            model_name=state.get("detector_model_name"),
        )
    except ServiceBusyError:
        logger.warning("[503] output_detection busy")
        raise
    except Exception as exc:  # noqa: BLE001
        raise DetectionError(f"输出检测失败: {exc}") from exc

    detection = _detection_payload(result)
    _emit(state, "output_detection", detection)

    if not result.is_safe:
        logger.warning(
            "[输出拦截] stopped_at=output_detection（命中类别: %s / 共 %s 条规则）",
            ", ".join(result.failed_rules) or "未给出",
            result.total_rules,
        )
        return {"output_detection": detection, "is_safe": False, "stopped_at": "output_detection"}
    return {"output_detection": detection, "is_safe": True}


# ── 条件路由 ────────────────────────────────────────────────────────


def route_after_input(state: PipelineState) -> str:
    """输入不安全 -> 直接结束；否则进入 RAG 阶段。"""
    return "rag" if (state.get("input_detection") or {}).get("is_safe") else "stop"


def route_after_policy(state: PipelineState) -> str:
    """已给出终止原因（跳过 / 取消）-> 结束；否则执行输出检测。"""
    return "stop" if state.get("stopped_at") else "output_detection"


# ── 图构建 / Agent 封装 ─────────────────────────────────────────────


def _retry_policy(steps: PipelineSteps):
    """检测节点的重试策略；``None`` 表示不重试。

    - 只重试 ``DetectionError``（下游异常 / 检测链路内部错误，多为瞬时的连接或超时）；
    - ``ServiceBusyError`` 是「并发名额没抢到，请稍后重试」的快速失败语义（-> 503），
      重试只会把它放大成重试风暴，故显式排除。
    """
    max_attempts = int(getattr(getattr(steps, "settings", None), "detect_node_max_attempts", 2) or 1)
    if max_attempts <= 1:
        return None

    from langgraph.types import RetryPolicy

    return RetryPolicy(
        max_attempts=max_attempts,
        initial_interval=0.3,
        backoff_factor=2.0,
        retry_on=DetectionError,
    )


def build_defense_graph(steps: PipelineSteps):
    """构建并编译 LangGraph 状态图。"""
    from langgraph.graph import END, START, StateGraph

    def bind(node: Callable[..., PipelineState]) -> Callable[[PipelineState, RunnableConfig], PipelineState]:
        def runner(state: PipelineState, config: RunnableConfig) -> PipelineState:
            return node(state, steps, config)

        runner.__name__ = node.__name__
        return runner

    builder = StateGraph(PipelineState)
    retry_policy = _retry_policy(steps)
    builder.add_node("input_detection", bind(input_detection_node), retry_policy=retry_policy)
    builder.add_node("rag_retrieval", bind(rag_retrieval_node))
    builder.add_node("generation", bind(generation_node))
    builder.add_node("skip_policy", bind(skip_policy_node))
    builder.add_node("output_detection", bind(output_detection_node), retry_policy=retry_policy)

    builder.add_edge(START, "input_detection")
    builder.add_conditional_edges("input_detection", route_after_input, {"stop": END, "rag": "rag_retrieval"})
    builder.add_edge("rag_retrieval", "generation")
    builder.add_edge("generation", "skip_policy")
    builder.add_conditional_edges(
        "skip_policy", route_after_policy, {"stop": END, "output_detection": "output_detection"}
    )
    builder.add_edge("output_detection", END)
    return builder.compile()


class DefenseAgent:
    """多阶段检测流水线的 Agent 封装（LangGraph 状态图）。"""

    def __init__(self, steps: PipelineSteps) -> None:
        self._steps = steps
        self.graph = build_defense_graph(steps)

    def run(self, payload: Dict[str, Any]) -> PipelineState:
        """同步执行整条流水线，返回最终状态。"""
        return self.graph.invoke(dict(payload))

    def stream(self, payload: Dict[str, Any], cancel_event: Optional[Any] = None) -> Iterator[Tuple[str, Any]]:
        """流式执行流水线，逐事件 yield ``(event, data)``。

        自定义流负责 ``input_detection`` / ``rag`` / ``delta`` / ``output_detection``，
        收尾由本方法在状态图结束后补一个 ``done`` 事件。``cancel_event`` 置位后，
        生成节点会在下一个增量处提前结束（用于客户端断流）。
        """
        config = {"configurable": {"cancel_event": cancel_event}}
        final: Optional[PipelineState] = None
        for mode, chunk in self.graph.stream(dict(payload), config=config, stream_mode=["custom", "values"]):
            if mode == "custom":
                yield chunk["event"], chunk["data"]
            elif mode == "values":
                final = chunk

        state: PipelineState = final if final is not None else dict(payload)
        yield (
            "done",
            {
                "is_safe": bool(state.get("is_safe", False)),
                "stopped_at": state.get("stopped_at"),
                "model_response": state.get("model_response"),
            },
        )
