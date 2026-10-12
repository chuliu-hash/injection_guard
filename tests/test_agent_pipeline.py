"""LangGraph 编排层单测：图结构、条件路由、节点行为（不加载模型）。"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from injection_guard.agent.pipeline import (
    DefenseAgent,
    build_defense_graph,
    generation_node,
    input_detection_node,
    route_after_input,
    route_after_policy,
    skip_policy_node,
)
from injection_guard.errors import DetectionError


class _Steps:
    """最小桩：只实现节点会调用的方法。"""

    def __init__(self, *, safe=True, chunks=("A", "B"), rag=None, output_safe=True):
        self._safe = safe
        self._chunks = list(chunks)
        self._rag = rag
        self._output_safe = output_safe
        self.output_calls = []

    def _run_input_detection_guarded(self, **_kw):
        return {"is_safe": self._safe, "p_safe": 0.99, "detection_error": None}

    def _run_rag(self, user_input):
        if self._rag is None:
            raise RuntimeError("embed down")
        return self._rag

    def _run_chat_guarded(self, prompt, max_tokens=None, temperature=None):
        self.output_calls.append({"chat_prompt": prompt})
        return "".join(self._chunks)

    def _run_chat_stream_guarded(self, prompt, max_tokens=None, temperature=None):
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


def test_agent_run_degrades_rag_failure():
    outcome = DefenseAgent(_Steps(rag=None)).run(_base_state(use_rag=True))
    assert outcome["rag"] is None
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
