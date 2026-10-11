"""基于 LangGraph 的防御流水线编排（企业级 Agent 工作流）。

状态图结构（条件边实现提前拦截 / 跳过）::

    START → input_detection ─┬─(不安全)→ END
                             └─(安全)→ rag_retrieval → generation → skip_policy
                                                    ├─(高置信)→ END
                                                    └──────→ output_detection → END

各节点只做「编排」：真正的检测 / 检索 / 生成仍由
``DefenseService`` 的并发受控方法（``_run_*_guarded``）执行，因此并发闸门、
RAG 降级、异常分类（``ServiceBusyError`` / ``DetectionError``）语义与旧实现一致。

- 同步模式（``stream=False``）：``DefenseAgent.run`` 用 ``graph.invoke`` 一次跑完；
- 流式模式（``stream=True``）：``DefenseAgent.stream`` 用
  ``graph.stream(stream_mode=["custom", "values"])`` 逐节点推送事件，节点内部经
  ``get_stream_writer()`` 发送 SSE 事件。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterator, Optional, Protocol, Tuple

from langchain_core.runnables import RunnableConfig

from ..errors import DetectionError, ServiceBusyError
from .state import CANCELLED_STOP, PipelineState

logger = logging.getLogger(__name__)


class PipelineSteps(Protocol):
    """流水线各阶段的实际执行者（由 ``DefenseService`` 实现）。"""

    def _run_input_detection_guarded(self, **kwargs: Any) -> Dict[str, Any]: ...

    def _run_output_detection_guarded(self, **kwargs: Any) -> Any: ...

    def _run_rag(self, user_input: str) -> Dict[str, Any]: ...

    def _run_chat_guarded(
        self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> str: ...

    def _run_chat_stream_guarded(
        self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> Iterator[str]: ...


def _emit(state: PipelineState, event: str, data: Any) -> None:
    """流式模式下通过 LangGraph 的自定义流推送一个 SSE 事件。"""
    if not state.get("stream"):
        return
    from langgraph.config import get_stream_writer

    get_stream_writer()({"event": event, "data": data})


def _cancel_event(config: Optional[Dict[str, Any]]):
    """取出编排层注入的取消信号（客户端断流置位）。"""
    configurable = (config or {}).get("configurable") or {}
    return configurable.get("cancel_event")


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
    """阶段 2：可选的 RAG 安全上下文增强（失败降级为原始输入）。"""
    rag_info: Optional[Dict[str, Any]] = None
    final_prompt = state["user_input"]

    if state.get("use_rag"):
        try:
            rag_info = steps._run_rag(state["user_input"])
            final_prompt = rag_info["final_prompt"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[RAG] 增强失败，降级为原始输入: %s", exc)
            rag_info = None
            final_prompt = state["user_input"]

    _emit(
        state,
        "rag",
        rag_info or {"enabled": False, "final_prompt": None, "retrieved_contexts": []},
    )
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

    if not state.get("stream"):
        try:
            return {"model_response": steps._run_chat_guarded(prompt, max_tokens=max_tokens, temperature=temperature)}
        except ServiceBusyError:
            logger.warning("[503] chat busy")
            raise
        except Exception as exc:  # noqa: BLE001
            raise DetectionError(f"聊天生成失败: {exc}") from exc

    cancel_event = _cancel_event(config)
    chunks: list[str] = []
    upstream = steps._run_chat_stream_guarded(prompt, max_tokens=max_tokens, temperature=temperature)
    try:
        for delta in upstream:
            if cancel_event is not None and cancel_event.is_set():
                # 客户端已断流：就地结束状态图，避免继续输出检测
                logger.info("[流式] 客户端断开，提前终止聊天生成")
                return {
                    "model_response": "".join(chunks),
                    "is_safe": False,
                    "stopped_at": CANCELLED_STOP,
                }
            chunks.append(delta)
            _emit(state, "delta", {"content": delta})
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

    return {"model_response": "".join(chunks)}


def skip_policy_node(
    state: PipelineState, steps: Optional[PipelineSteps] = None, config: Optional[Dict[str, Any]] = None
) -> PipelineState:
    """阶段 4：输入高置信时跳过输出检测（仍保留生成的回复）。"""
    if not state.get("skip_output_when_high_confidence"):
        return {}
    input_detection = state.get("input_detection") or {}
    if input_detection.get("p_safe", 0.0) >= state.get("safe_skip_threshold", 1.0):
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

    detection = {
        "is_safe": result.is_safe,
        "failed_rules": result.failed_rules,
        "total_rules": result.total_rules,
    }
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


def build_defense_graph(steps: PipelineSteps):
    """构建并编译 LangGraph 状态图。"""
    from langgraph.graph import END, START, StateGraph

    def bind(node: Callable[..., PipelineState]) -> Callable[[PipelineState, RunnableConfig], PipelineState]:
        def runner(state: PipelineState, config: RunnableConfig) -> PipelineState:
            return node(state, steps, config)

        runner.__name__ = node.__name__
        return runner

    builder = StateGraph(PipelineState)
    builder.add_node("input_detection", bind(input_detection_node))
    builder.add_node("rag_retrieval", bind(rag_retrieval_node))
    builder.add_node("generation", bind(generation_node))
    builder.add_node("skip_policy", bind(skip_policy_node))
    builder.add_node("output_detection", bind(output_detection_node))

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
