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
        output_detect_interval_tokens=0,
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
            is_safe=True, failed_rules=[], total_rules=12
        )
        self._rag_result = rag_result
        self._rag_exc = rag_exc
        self._chat_chunks = list(chat_chunks)
        self.chat_prompts = []
        self.chat_params = []
        self.chat_history = []
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

    def _run_chat_stream_guarded(self, prompt, chat_options=None, history=None):
        self.chat_prompts.append(prompt)
        self.chat_params.append(chat_options)
        self.chat_history.append(history)
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


def test_chat_generation_params_passed_through():
    """流式路径同样把请求体的生成参数透传给聊天客户端。"""
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    _collect(svc, user_input="你好", temperature=1.2, max_tokens=256)
    assert svc.chat_params == [{"max_tokens": 256, "temperature": 1.2}]


def test_request_skip_flag_enables_output_skip():
    """流式路径：请求体 skip_output_when_high_confidence=True 覆盖 Settings 的 false。"""
    svc = _StubService(
        _settings(skip_output_when_high_confidence=False), input_result=SAFE_INPUT
    )
    events = _collect(svc, user_input="你好", skip_output_when_high_confidence=True)
    assert events[-1][1]["stopped_at"] == "output_skipped"
    assert svc.output_calls == []


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
    assert svc.chat_history == [None]  # 没给历史就是单轮


def test_history_is_forwarded_to_streaming_chat():
    """多轮记忆：流式生成同样把历史转给聊天客户端。"""
    history = [
        {"role": "user", "content": "第一问"},
        {"role": "assistant", "content": "第一答"},
    ]
    svc = _StubService(_settings(), input_result=SAFE_INPUT)
    _collect(svc, user_input="第二问", history=history)
    assert svc.chat_history == [history]


def test_rag_disabled_by_default_emits_disabled_event():
    svc = _StubService(_settings(enable_rag=False), input_result=SAFE_INPUT)
    events = _collect(svc, user_input="你好")
    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["enabled"] is False
    assert rag_event["degraded"] is False
    assert rag_event["reason"] == "disabled"
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
    # 与「未启用」区分：开了但检索失败
    assert rag_event["enabled"] is True
    assert rag_event["degraded"] is True
    assert rag_event["reason"] == "error"
    assert "embed down" in rag_event["error"]
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


def test_stream_incremental_output_detection_intercepts_mid_generation():
    """每累计约 512 token 检测一次：命中不安全立刻中断生成并推 output_detection。"""
    svc = _StubService(
        _settings(output_detect_interval_tokens=512),
        input_result=SAFE_INPUT,
        output_result=SimpleNamespace(is_safe=False, failed_rules=["S5"], total_rules=12),
        chat_chunks=("安" * 600, "全" * 600),
    )
    events = _collect(svc, user_input="你好")

    assert _names(events) == ["input_detection", "rag", "delta", "output_detection", "done"]
    assert len(svc.output_calls) == 1
    assert svc.output_calls[0]["answer"] == "安" * 600
    assert events[-1][0] == "done"
    assert events[-1][1]["is_safe"] is False
    assert events[-1][1]["stopped_at"] == "output_detection"
    assert events[-1][1]["model_response"] == "安" * 600


def test_stream_incremental_output_detection_keeps_generating_when_safe():
    """增量检测都安全：仍会生成完整回复，收尾再做一次整段检测。"""
    svc = _StubService(
        _settings(output_detect_interval_tokens=512),
        input_result=SAFE_INPUT,
        chat_chunks=("安" * 600, "安" * 600, "尾"),
    )
    events = _collect(svc, user_input="你好")

    assert _names(events) == [
        "input_detection",
        "rag",
        "delta",
        "delta",
        "delta",
        "output_detection",
        "done",
    ]
    assert len(svc.output_calls) == 3   # 2 次增量 + 1 次收尾
    assert events[-1][1]["model_response"] == "安" * 1200 + "尾"


def test_stream_incremental_detection_disabled_when_interval_zero():
    svc = _StubService(_settings(output_detect_interval_tokens=0), input_result=SAFE_INPUT)
    events = _collect(svc, user_input="你好")
    assert _names(events) == [
        "input_detection",
        "rag",
        "delta",
        "delta",
        "output_detection",
        "done",
    ]
    assert len(svc.output_calls) == 1   # 只有收尾那一次


def test_output_unsafe_marks_done_stopped_at_output_detection():
    """输出不安全：delta 已发完，done 以 stopped_at="output_detection" 收尾。"""
    svc = _StubService(
        _settings(),
        input_result=SAFE_INPUT,
        output_result=SimpleNamespace(
            is_safe=False, failed_rules=["S5"], total_rules=12
        ),
    )
    events = _collect(svc, user_input="你好")

    # delta 仍在 output_detection 之前全部发出（这是「先流式」的固有语义）
    assert _names(events) == [
        "input_detection",
        "rag",
        "delta",
        "delta",
        "output_detection",
        "done",
    ]
    assert events[-2][1]["is_safe"] is False
    assert events[-1][0] == "done"
    assert events[-1][1]["is_safe"] is False
    assert events[-1][1]["stopped_at"] == "output_detection"
    assert events[-1][1]["model_response"] == "REPLY"


def test_rag_no_match_skips_context_and_uses_raw_input():
    """检索没命中：rag 事件标 no_match，且不拼接上下文（直接用原始输入）。"""
    rag = {
        "enabled": True,
        "final_prompt": None,
        "reason": "no_match",
        "retrieved_contexts": [],
    }
    svc = _StubService(_settings(enable_rag=True), input_result=SAFE_INPUT, rag_result=rag)
    events = _collect(svc, user_input="你好")

    rag_event = next(d for name, d in events if name == "rag")
    assert rag_event["enabled"] is True
    assert rag_event["degraded"] is False
    assert rag_event["reason"] == "no_match"
    assert rag_event["final_prompt"] is None
    assert svc.chat_prompts == ["你好"]  # 退回原始输入，而不是拼上空上下文
