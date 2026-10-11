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
