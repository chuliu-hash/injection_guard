"""P1 请求追踪单测（不依赖 OpenTelemetry SDK）。"""

from __future__ import annotations

from injection_guard.observability import (
    new_trace,
    record_node,
    trace_from_config,
    trace_payload,
)


def test_trace_records_nodes_and_serializes():
    trace = new_trace(thread_id="thread-1", trace_id="trace-1")
    started = trace._started_perf
    record_node(trace, "input_detection", started)
    record_node(trace, "generation", started, status="error", error=ValueError("boom"))

    payload = trace_payload(trace)

    assert payload["trace_id"] == "trace-1"
    assert payload["thread_id"] == "thread-1"
    assert [node["name"] for node in payload["nodes"]] == ["input_detection", "generation"]
    assert payload["nodes"][1]["status"] == "error"
    assert "boom" in payload["nodes"][1]["error"]
    assert payload["duration_ms"] >= 0


def test_trace_from_config_returns_none_for_other_values():
    trace = new_trace()
    assert trace_from_config({"configurable": {"trace": trace}}) is trace
    assert trace_from_config({"configurable": {"trace": "not-a-trace"}}) is None
    assert trace_from_config(None) is None
