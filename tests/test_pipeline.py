"""DefenseService.detect 端到端编排测试（全部用桩，不加载模型、不联网）。"""

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
        chat_reply="REPLY",
    ):
        self.settings = settings
        self._input_result = input_result
        self._output_result = output_result or SimpleNamespace(
            is_safe=True, failed_rules=[], total_rules=12
        )
        self._rag_result = rag_result
        self._rag_exc = rag_exc
        self._chat_reply = chat_reply
        self.chat_prompts = []
        self.chat_params = []
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

    def _run_chat_guarded(self, prompt, max_tokens=None, temperature=None):
        self.chat_prompts.append(prompt)
        self.chat_params.append({"max_tokens": max_tokens, "temperature": temperature})
        return self._chat_reply


def test_input_unsafe_stops_and_skips_chat():
    svc = _StubService(_settings(), input_result=UNSAFE_INPUT)
    outcome = svc.detect(user_input="攻击")
    assert outcome.is_safe is False
    assert outcome.stopped_at == "input_detection"
    assert outcome.output_detection is None
    assert svc.chat_prompts == []


def test_generates_response_when_not_provided():
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    outcome = svc.detect(user_input="你好")
    assert svc.chat_prompts == ["你好"]
    assert outcome.model_response == "REPLY"
    assert outcome.stopped_at is None


def test_chat_generation_params_passed_through():
    """请求体的 temperature / max_tokens 透传给聊天客户端。"""
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    svc.detect(user_input="你好", temperature=0.3, max_tokens=64)
    assert svc.chat_params == [{"max_tokens": 64, "temperature": 0.3}]


def test_chat_generation_params_default_none():
    """未提供时传 None，由 ChatClient 回落到 Settings 默认值。"""
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    svc.detect(user_input="你好")
    assert svc.chat_params == [{"max_tokens": None, "temperature": None}]


def test_request_skip_flag_enables_output_skip():
    """请求体 skip_output_when_high_confidence=True 覆盖 Settings 的 false。"""
    svc = _StubService(
        _settings(skip_output_when_high_confidence=False), input_result=SAFE_INPUT
    )
    outcome = svc.detect(user_input="你好", skip_output_when_high_confidence=True)
    assert outcome.stopped_at == "output_skipped"
    assert svc.output_calls == []


def test_request_threshold_controls_output_skip():
    """请求体 safe_skip_threshold 覆盖 Settings：0.99 < 0.999 → 不跳过。"""
    svc = _StubService(
        _settings(skip_output_when_high_confidence=True, safe_skip_threshold=0.5),
        input_result=SAFE_INPUT,
    )
    outcome = svc.detect(user_input="你好", safe_skip_threshold=0.999)
    assert outcome.stopped_at is None
    assert len(svc.output_calls) == 1


def test_provided_response_skips_chat():
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    outcome = svc.detect(user_input="你好", model_response="已有的回复")
    assert svc.chat_prompts == []
    assert outcome.model_response == "已有的回复"


def test_rag_enabled_feeds_final_prompt_to_chat():
    rag = {"enabled": True, "final_prompt": "FINAL", "retrieved_contexts": []}
    svc = _StubService(_settings(enable_rag=True), input_result=SAFE_INPUT, rag_result=rag)
    outcome = svc.detect(user_input="你好")
    assert svc.chat_prompts == ["FINAL"]
    assert outcome.rag["final_prompt"] == "FINAL"


def test_rag_failure_degrades_to_original_input():
    svc = _StubService(
        _settings(enable_rag=True), input_result=SAFE_INPUT, rag_exc=RuntimeError("embed down")
    )
    outcome = svc.detect(user_input="你好")
    assert svc.chat_prompts == ["你好"]
    assert outcome.rag is None


def test_rag_disabled_by_default():
    svc = _StubService(_settings(enable_rag=False), input_result=SAFE_INPUT)
    outcome = svc.detect(user_input="你好")
    assert outcome.rag is None
    assert svc.chat_prompts == ["你好"]


def test_request_level_use_rag_overrides_global():
    rag = {"enabled": True, "final_prompt": "FINAL", "retrieved_contexts": []}
    svc = _StubService(_settings(enable_rag=False), input_result=SAFE_INPUT, rag_result=rag)
    outcome = svc.detect(user_input="你好", use_rag=True)
    assert outcome.rag is not None
    assert svc.chat_prompts == ["FINAL"]


def test_high_confidence_skips_output_detection():
    svc = _StubService(
        _settings(skip_output_when_high_confidence=True), input_result=SAFE_INPUT
    )
    outcome = svc.detect(user_input="你好")
    assert outcome.stopped_at == "output_skipped"
    assert outcome.output_detection is None
    assert outcome.model_response == "REPLY"


def test_output_detection_uses_original_user_input():
    rag = {"enabled": True, "final_prompt": "FINAL", "retrieved_contexts": []}
    svc = _StubService(_settings(enable_rag=True), input_result=SAFE_INPUT, rag_result=rag)
    outcome = svc.detect(user_input="原始输入")
    assert outcome.stopped_at is None
    assert svc.output_calls[0]["text"] == "原始输入"
    assert svc.output_calls[0]["answer"] == "REPLY"


def test_output_unsafe_marks_stopped_at_output_detection():
    """输出检测判不安全 -> stopped_at="output_detection"，model_response 原样返回。"""
    svc = _StubService(
        _settings(),
        input_result=SAFE_INPUT,
        output_result=SimpleNamespace(
            is_safe=False, failed_rules=["S5"], total_rules=12
        ),
    )
    outcome = svc.detect(user_input="你好")
    assert outcome.is_safe is False
    assert outcome.stopped_at == "output_detection"
    assert outcome.model_response == "REPLY"
    assert outcome.output_detection["failed_rules"] == ["S5"]


def test_output_safe_keeps_stopped_at_none():
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    outcome = svc.detect(user_input="你好")
    assert outcome.is_safe is True
    assert outcome.stopped_at is None
