"""LLM 调用指标（本地可观测性，不依赖 LangSmith 等外部服务）。

给 ChatModel 挂上 :class:`MetricsCallbackHandler` 后，每次调用的耗时与 token 用量会：
1. 打一条日志（``label`` 区分聊天 / 输入复审 / 输出检测）；
2. 累加到进程级计数器，供 ``snapshot()`` 查询（``GET /health`` 会带上）。

指标是**进程内**的（重启归零、多 worker 各自统计），够用于排障与容量估算；
需要跨进程 / 跨请求的追踪时应改接 LangSmith 等外部方案。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


@dataclass
class _Counter:
    """单个 label 的累计指标。"""

    calls: int = 0
    failures: int = 0
    latency_ms_total: float = 0.0
    latency_ms_max: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_LOCK = threading.Lock()
_COUNTERS: Dict[str, _Counter] = {}


def _counter(label: str) -> _Counter:
    return _COUNTERS.setdefault(label, _Counter())


def record_call(
    label: str,
    *,
    latency_ms: float,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    failed: bool = False,
) -> None:
    """累加一次调用（供回调使用；也可在事务边界手工记录）。"""
    with _LOCK:
        counter = _counter(label)
        counter.calls += 1
        if failed:
            counter.failures += 1
        counter.latency_ms_total += max(0.0, latency_ms)
        counter.latency_ms_max = max(counter.latency_ms_max, latency_ms)
        counter.prompt_tokens += max(0, prompt_tokens)
        counter.completion_tokens += max(0, completion_tokens)


def snapshot() -> Dict[str, Dict[str, Any]]:
    """返回各 label 的指标快照（含平均耗时与总 token 数）。"""
    with _LOCK:
        return {
            label: {
                "calls": counter.calls,
                "failures": counter.failures,
                "avg_latency_ms": (
                    round(counter.latency_ms_total / counter.calls, 1) if counter.calls else 0.0
                ),
                "max_latency_ms": round(counter.latency_ms_max, 1),
                "prompt_tokens": counter.prompt_tokens,
                "completion_tokens": counter.completion_tokens,
            }
            for label, counter in _COUNTERS.items()
        }


def reset() -> None:
    """清空全部指标（测试用）。"""
    with _LOCK:
        _COUNTERS.clear()


def _usage_from_response(response: Any) -> Tuple[int, int]:
    """从 ``LLMResult`` 里取 ``(prompt_tokens, completion_tokens)``，取不到就返回 0。"""
    output = getattr(response, "llm_output", None) or {}
    usage = output.get("token_usage") or output.get("usage") or {}
    if usage:
        return (
            int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )

    # OpenAI 兼容服务有时只把用量挂在消息的 usage_metadata 上
    for generations in getattr(response, "generations", None) or []:
        for generation in generations:
            message = getattr(generation, "message", None)
            metadata = getattr(message, "usage_metadata", None) or {}
            if metadata:
                return (
                    int(metadata.get("input_tokens") or 0),
                    int(metadata.get("output_tokens") or 0),
                )
    return 0, 0


class MetricsCallbackHandler(BaseCallbackHandler):
    """把每次 LLM 调用的耗时 / token 用量记入进程级计数器。"""

    def __init__(self, label: str = "llm") -> None:
        self.label = label
        self._started_at: Dict[str, float] = {}
        self._lock = threading.Lock()

    # ── 回调钩子 ─────────────────────────────────────────────────────
    def on_llm_start(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_chat_model_start(self, *args: Any, run_id: Any = None, **kwargs: Any) -> None:
        self._mark_start(run_id)

    def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        latency_ms = self._elapsed_ms(run_id)
        prompt_tokens, completion_tokens = _usage_from_response(response)
        record_call(
            self.label,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        logger.info(
            "[LLM指标] %s 用时=%.0fms tokens=in:%d/out:%d",
            self.label,
            latency_ms,
            prompt_tokens,
            completion_tokens,
        )

    def on_llm_error(self, error: BaseException, *, run_id: Any = None, **kwargs: Any) -> None:
        latency_ms = self._elapsed_ms(run_id)
        record_call(self.label, latency_ms=latency_ms, failed=True)
        logger.warning("[LLM指标] %s 调用失败 用时=%.0fms: %s", self.label, latency_ms, error)

    # ── 内部 ─────────────────────────────────────────────────────────
    def _mark_start(self, run_id: Any) -> None:
        with self._lock:
            self._started_at[str(run_id)] = time.perf_counter()

    def _elapsed_ms(self, run_id: Any) -> float:
        with self._lock:
            started = self._started_at.pop(str(run_id), None)
        if started is None:
            return 0.0
        return (time.perf_counter() - started) * 1000.0
