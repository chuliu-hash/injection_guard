"""轻量请求追踪（P1）。

不依赖外部服务即可工作：每次流水线执行生成一个 ``trace_id``，记录每个 LangGraph
节点的耗时与状态，并随响应 / SSE ``done`` 事件返回。若运行环境安装了
OpenTelemetry SDK，节点执行会额外产生 span；未安装时自动退化为纯内存追踪。

设计目标：

- 追踪对象只挂在 LangGraph 的 ``configurable`` 上，不写入业务状态，避免污染
  ``PipelineState`` 或影响检查点内容；
- 不记录用户原文 / 模型原文，只记录节点名、耗时、状态和可选属性，降低日志泄露风险；
- 同步与流式路径共用同一套实现。
"""

from __future__ import annotations

import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class NodeTrace:
    """单个节点的执行记录。"""

    name: str
    duration_ms: float
    status: str = "ok"
    error: Optional[str] = None


@dataclass
class RequestTrace:
    """一次流水线执行的追踪上下文。"""

    trace_id: str
    thread_id: str
    started_at: float
    _started_perf: float = field(repr=False)
    nodes: List[NodeTrace] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)

    def finish(self) -> None:
        self.attributes.setdefault(
            "duration_ms", round((time.perf_counter() - self._started_perf) * 1000.0, 3)
        )


def new_trace(thread_id: Optional[str] = None, trace_id: Optional[str] = None) -> RequestTrace:
    """创建追踪上下文；``thread_id`` 缺省时生成 UUID。"""
    return RequestTrace(
        trace_id=trace_id or str(uuid.uuid4()),
        thread_id=thread_id or str(uuid.uuid4()),
        started_at=time.time(),
        _started_perf=time.perf_counter(),
    )


def trace_from_config(config: Optional[Dict[str, Any]]) -> Optional[RequestTrace]:
    """从 LangGraph ``RunnableConfig`` 取出追踪上下文。"""
    configurable = (config or {}).get("configurable") or {}
    trace = configurable.get("trace")
    return trace if isinstance(trace, RequestTrace) else None


def record_node(
    trace: Optional[RequestTrace],
    name: str,
    started_perf: float,
    *,
    status: str = "ok",
    error: Optional[BaseException] = None,
) -> None:
    """记录一次节点执行（``trace`` 为空时静默忽略）。"""
    if trace is None:
        return
    trace.nodes.append(
        NodeTrace(
            name=name,
            duration_ms=round((time.perf_counter() - started_perf) * 1000.0, 3),
            status=status,
            error=(str(error)[:300] if error is not None else None),
        )
    )


def trace_payload(trace: Optional[RequestTrace]) -> Optional[Dict[str, Any]]:
    """转换成可 JSON 序列化的结构。"""
    if trace is None:
        return None
    trace.finish()
    return {
        "trace_id": trace.trace_id,
        "thread_id": trace.thread_id,
        "started_at": trace.started_at,
        "duration_ms": trace.attributes.get("duration_ms"),
        "nodes": [
            {
                "name": node.name,
                "duration_ms": node.duration_ms,
                "status": node.status,
                **({"error": node.error} if node.error else {}),
            }
            for node in trace.nodes
        ],
        **({"attributes": trace.attributes} if trace.attributes else {}),
    }


@contextlib.contextmanager
def node_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    """可选的 OpenTelemetry span；无 SDK / 无 exporter 时为 no-op。"""
    try:
        from opentelemetry import trace as otel_trace
    except Exception:  # noqa: BLE001 - 可观测性不得影响主流程
        yield
        return

    try:
        tracer = otel_trace.get_tracer("injection_guard")
        span_cm = tracer.start_as_current_span(name)
    except Exception:  # noqa: BLE001 - exporter / SDK 初始化异常时退化为 no-op
        yield
        return

    with span_cm as span:
        for key, value in (attributes or {}).items():
            if value is not None:
                span.set_attribute(key, value)
        yield
