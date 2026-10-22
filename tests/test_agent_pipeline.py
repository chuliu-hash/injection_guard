"""LangGraph 编排层单测：图结构、条件路由、节点行为（不加载模型）。"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver

from injection_guard.agent.pipeline import (
    DefenseAgent,
    build_defense_graph,
    generation_node,
    input_detection_node,
    route_after_input,
    route_after_policy,
    skip_policy_node,
)
from injection_guard.errors import DetectionError, ServiceBusyError


class _Steps:
    """最小桩：只实现节点会调用的方法。"""

    def __init__(self, *, safe=True, chunks=("A", "B"), rag=None, output_safe=True):
        self._safe = safe
        self._chunks = list(chunks)
        self._rag = rag
        self._output_safe = output_safe
        self.chat_calls = []
        self.output_calls = []

    def _run_input_detection_guarded(self, **_kw):
        return {"is_safe": self._safe, "p_safe": 0.99, "detection_error": None}

    def _run_rag(self, user_input):
        if self._rag is None:
            raise RuntimeError("embed down")
        return self._rag

    def _run_chat_guarded(self, prompt, chat_options=None, history=None):
        self.chat_calls.append(
            {"prompt": prompt, "history": history, "chat_options": chat_options}
        )
        return "".join(self._chunks)

    def _run_chat_stream_guarded(self, prompt, chat_options=None, history=None):
        self.chat_calls.append(
            {"prompt": prompt, "history": history, "chat_options": chat_options}
        )
        yield from self._chunks

    def _run_output_detection_guarded(self, **kw):
        self.output_calls.append(kw)
        return SimpleNamespace(
            is_safe=self._output_safe, failed_rules=[] if self._output_safe else ["S1"], total_rules=12
        )


def _base_state(**over):
    state = {
        "user_input": "hi",
        "model_response": None,
        "use_rag": False,
        "stream": False,
        "skip_output_when_high_confidence": False,
        "safe_skip_threshold": 0.5,
    }
    state.update(over)
    return state


def test_graph_exposes_pipeline_stages():
    graph = build_defense_graph(_Steps())
    assert set(graph.get_graph().nodes) >= {
        "__start__",
        "input_detection",
        "rag_retrieval",
        "generation",
        "skip_policy",
        "output_detection",
    }


def test_routes_stop_when_unsafe_or_already_terminated():
    assert route_after_input({"input_detection": {"is_safe": False}}) == "stop"
    assert route_after_input({"input_detection": {"is_safe": True}}) == "rag"
    assert route_after_policy({}) == "output_detection"
    assert route_after_policy({"stopped_at": "output_skipped"}) == "stop"


def test_input_node_marks_unsafe_and_intercepts():
    node_out = input_detection_node(_base_state(), _Steps(safe=False))
    assert node_out["is_safe"] is False
    assert node_out["stopped_at"] == "input_detection"


def test_input_node_raises_on_detection_error():
    class _ErrSteps(_Steps):
        def _run_input_detection_guarded(self, **_kw):
            return {"is_safe": True, "detection_error": "model crashed"}

    try:
        input_detection_node(_base_state(), _ErrSteps())
    except DetectionError as exc:
        assert "model crashed" in str(exc)
    else:  # pragma: no cover - 必须抛错
        raise AssertionError("detection_error 应转成 DetectionError")


def test_generation_node_skips_chat_when_response_provided():
    out = generation_node(_base_state(model_response="已有回复"), _Steps())
    assert out == {}


def test_chat_options_use_runtime_config_not_checkpointed_state():
    """API Key 等请求配置只走 runtime config，不写进 LangGraph 状态。"""
    steps = _Steps()
    options = {"api_url": "https://example.com/v1", "api_key": "secret", "model": "m"}

    final = DefenseAgent(steps).run(_base_state(), chat_options=options)

    assert steps.chat_calls[0]["chat_options"] == options
    assert "chat_options" not in final
    assert "secret" not in repr(final)


def test_chat_api_key_is_not_written_to_checkpointer():
    saver = MemorySaver()
    steps = _Steps()
    agent = DefenseAgent(steps, checkpointer=saver)

    agent.run(
        _base_state(),
        thread_id="secret-thread",
        chat_options={"api_key": "super-secret-key"},
    )

    assert "super-secret-key" not in repr(vars(saver))


def test_agent_run_degrades_rag_failure():
    outcome = DefenseAgent(_Steps(rag=None)).run(_base_state(use_rag=True))
    assert outcome["rag"]["enabled"] is True
    assert outcome["rag"]["degraded"] is True
    assert outcome["rag"]["reason"] == "error"
    assert outcome["final_prompt"] == "hi"
    assert outcome["is_safe"] is True


def test_agent_stream_emits_events_and_done():
    events = list(DefenseAgent(_Steps()).stream(_base_state(stream=True)))
    assert [name for name, _ in events] == [
        "input_detection",
        "rag",
        "delta",
        "delta",
        "output_detection",
        "done",
    ]
    assert events[-1][1]["model_response"] == "AB"


def test_agent_run_exposes_thread_id_and_trace():
    outcome = DefenseAgent(_Steps()).run(_base_state(), thread_id="thread-1")

    assert outcome["thread_id"] == "thread-1"
    assert outcome["trace"]["thread_id"] == "thread-1"
    assert [node["name"] for node in outcome["trace"]["nodes"]] == [
        "input_detection",
        "rag_retrieval",
        "generation",
        "skip_policy",
        "output_detection",
    ]


def test_agent_stream_done_exposes_thread_id_and_trace():
    events = list(DefenseAgent(_Steps()).stream(_base_state(stream=True), thread_id="thread-2"))
    done = events[-1][1]

    assert done["thread_id"] == "thread-2"
    assert done["trace"]["trace_id"]
    assert done["trace"]["thread_id"] == "thread-2"


def test_agent_accepts_checkpointer():
    agent = DefenseAgent(_Steps(), checkpointer=MemorySaver())
    outcome = agent.run(_base_state(), thread_id="thread-3")
    assert outcome["thread_id"] == "thread-3"


def test_agent_stream_cancel_skips_output_detection():
    steps = _Steps(chunks=("A", "B", "C", "D"))
    cancel = threading.Event()
    stream = DefenseAgent(steps).stream(_base_state(stream=True), cancel_event=cancel)

    emitted = []
    for event in stream:
        emitted.append(event[0])
        if event[0] == "rag":
            cancel.set()
    assert "output_detection" not in emitted
    assert steps.output_calls == []


def test_stream_generation_stops_at_interval_when_output_unsafe():
    """流式增量检测命中：立即中断生成，不再产出后续 delta。"""

    class _UnsafeOutput(_Steps):
        def _run_output_detection_guarded(self, **kw):
            self.output_calls.append(kw)
            return SimpleNamespace(is_safe=False, failed_rules=["S5"], total_rules=12)

    steps = _UnsafeOutput(chunks=("安" * 600, "全" * 600))
    state = _base_state(stream=True, output_detect_interval=512)
    out = generation_node(state, steps)

    assert out["stopped_at"] == "output_detection"
    assert out["is_safe"] is False
    assert out["model_response"] == "安" * 600      # 第二段没有生成
    assert len(steps.output_calls) == 1
    assert steps.output_calls[0]["answer"] == "安" * 600


def test_stream_generation_checks_each_interval_and_keeps_going():
    """增量检测判安全：继续生成，收尾由 output_detection 节点再做一次整段检测。"""
    steps = _Steps(chunks=("安" * 600, "安" * 600))
    out = generation_node(_base_state(stream=True, output_detect_interval=512), steps)

    assert out["model_response"] == "安" * 1200
    assert "stopped_at" not in out
    assert len(steps.output_calls) == 2


def test_stream_generation_skips_incremental_when_input_high_confidence():
    """输入高置信跳过输出检测时，流式过程中同样不做增量检测。"""
    steps = _Steps(chunks=("安" * 600, "安" * 600))
    state = _base_state(
        stream=True,
        output_detect_interval=512,
        skip_output_when_high_confidence=True,
        input_detection={"is_safe": True, "p_safe": 0.99},
    )
    out = generation_node(state, steps)

    assert out["model_response"] == "安" * 1200
    assert steps.output_calls == []


def test_skip_policy_does_not_override_upstream_stop():
    """上游已终止（增量检测命中）时，跳过策略不得覆盖终止原因。"""
    state = _base_state(
        stopped_at="output_detection",
        skip_output_when_high_confidence=True,
        input_detection={"is_safe": True, "p_safe": 0.99},
    )
    assert skip_policy_node(state) == {}


# ── 节点级重试策略（LangGraph RetryPolicy）───────────────────────────


class _FlakyInputSteps(_Steps):
    """输入检测固定失败，用于观察图是否会重试。"""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc
        self.input_calls = 0

    def _run_input_detection_guarded(self, **_kw):
        self.input_calls += 1
        raise self._exc


def test_detection_node_retries_internal_errors():
    steps = _FlakyInputSteps(DetectionError("输入检测失败: 下游抖动"))
    agent = DefenseAgent(steps)

    with pytest.raises(DetectionError):
        agent.run(_base_state())

    # DETECT_NODE_MAX_ATTEMPTS 默认 2：失败后重试一次
    assert steps.input_calls == 2


def test_detection_node_does_not_retry_busy_errors():
    """ServiceBusyError 是快速失败语义（-> 503），重试只会放大它。"""
    steps = _FlakyInputSteps(ServiceBusyError("输入检测服务繁忙"))
    agent = DefenseAgent(steps)

    with pytest.raises(ServiceBusyError):
        agent.run(_base_state())

    assert steps.input_calls == 1
