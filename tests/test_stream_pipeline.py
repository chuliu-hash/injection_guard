"""``DefenseService.stream_detect`` 事件序列测试（全部用桩，不加载模型、不联网）。"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.api.dependencies import DefenseService

SAFE_INPUT = {"is_safe": True, "p_safe": 0.99, "detection_error": None}
UNSAFE_INPUT = {"is_safe": False, "p_safe": 0.01, "detection_error": None}


def _settings(**over):
    base = dict(
        enable_rag=False,
        rag_top_k=2,
        skip_output_when_high_confidence=False,
        safe_skip_threshold=0.5,
        input_acquire_timeout=1,
        output_acquire_timeout=1,
        chat_acquire_timeout=1,
    )
    base.update(over)
    return SimpleNamespace(**base)


class _StubService(DefenseService):
    """跳过 torch/模型初始化，用桩替换各检测/生成步骤。"""

    def __init__(
        self,
        settings,
        *,
        input_result,
        output_result=None,
        rag_result=None,
        rag_exc=None,
        chat_chunks=("RE", "PLY"),
    ):
        self.settings = settings
        self._input_result = input_result
        self._output_result = output_result or SimpleNamespace(
            is_safe=True, failed_rules=[], total_rules=5
        )
        self._rag_result = rag_result
        self._rag_exc = rag_exc
        self._chat_chunks = list(chat_chunks)
        self.chat_prompts = []
        self.output_calls = []

    def _run_input_detection_guarded(self, **_kw):
        return self._input_result

    def _run_output_detection_guarded(self, **kw):
        self.output_calls.append(kw)
        return self._output_result

    def _run_rag(self, user_input):
        if self._rag_exc is not None:
            raise self._rag_exc
        return self._rag_result

    def _run_chat_stream_guarded(self, prompt):
        self.chat_prompts.append(prompt)
        yield from self._chat_chunks


def _collect(svc, **kw):
    return list(svc.stream_detect(**kw))


def _names(events):
    return [name for name, _ in events]


def test_input_unsafe_emits_input_then_done():
    svc = _StubService(_settings(), input_result=UNSAFE_INPUT)
    events = _collect(svc, user_input="攻击")
    assert _names(events) == ["input_detection", "done"]
    assert events[-1][1]["is_safe"] is False
    assert events[-1][1]["stopped_at"] == "input_detection"
    assert svc.chat_prompts == []


def test_event_sequence_happy_path():
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    events = _collect(svc, user_input="你好")
    assert _names(events) == [
        "input_detection",
        "rag",
        "delta",
        "delta",
        "output_detection",
        "done",
    ]
    assert "".join(d["content"] for name, d in events if name == "delta") == "REPLY"
    assert events[-1][1]["model_response"] == "REPLY"
    assert events[-1][1]["stopped_at"] is None
    assert svc.output_calls[0]["text"] == "你好"
    assert svc.output_calls[0]["answer"] == "REPLY"


def test_rag_disabled_by_default_emits_disabled_event():
    svc = _StubService(_settings(enable_rag=False), input_result=SAFE_INPUT)
    events = _collect(svc, user_input="你好")
    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["enabled"] is False
    assert svc.chat_prompts == ["你好"]


def test_rag_enabled_feeds_final_prompt_to_chat():
    rag = {"enabled": True, "final_prompt": "FINAL", "retrieved_contexts": []}
    svc = _StubService(_settings(enable_rag=True), input_result=SAFE_INPUT, rag_result=rag)
    events = _collect(svc, user_input="你好")
    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["final_prompt"] == "FINAL"
    assert svc.chat_prompts == ["FINAL"]


def test_rag_failure_degrades_to_original_input():
    svc = _StubService(
        _settings(enable_rag=True), input_result=SAFE_INPUT, rag_exc=RuntimeError("embed down")
    )
    events = _collect(svc, user_input="你好")
    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["enabled"] is False
    assert svc.chat_prompts == ["你好"]


def test_request_level_use_rag_overrides_global():
    rag = {"enabled": True, "final_prompt": "FINAL", "retrieved_contexts": []}
    svc = _StubService(_settings(enable_rag=False), input_result=SAFE_INPUT, rag_result=rag)
    events = _collect(svc, user_input="你好", use_rag=True)
    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["enabled"] is True
    assert svc.chat_prompts == ["FINAL"]


def test_high_confidence_skips_output_detection():
    svc = _StubService(
        _settings(skip_output_when_high_confidence=True), input_result=SAFE_INPUT
    )
    events = _collect(svc, user_input="你好")
    assert "output_detection" not in _names(events)
    assert events[-1][1]["stopped_at"] == "output_skipped"
    assert events[-1][1]["model_response"] == "REPLY"


def test_provided_response_skips_chat_and_delta():
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    events = _collect(svc, user_input="你好", model_response="已有回复")
    assert "delta" not in _names(events)
    assert svc.chat_prompts == []
    assert events[-1][1]["model_response"] == "已有回复"
