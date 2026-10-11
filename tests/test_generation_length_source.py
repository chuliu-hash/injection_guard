"""生成长度的单一来源：输入/输出检测的客户端不发送生成长度参数。

生成长度由推理服务按 ``--service`` 读取的 ``<服务>_API_SERVER_MAX_NEW_TOKENS`` 决定
（``llm_server._prepare_generation`` 取 min(请求上限, 服务端上限)）；客户端发送长度参数
仅保留给聊天（``CHAT_MAX_TOKENS``，LangChain 侧写作 ``max_completion_tokens``）。
这里不发起网络请求，只断言客户端/服务端的字段契约。
"""

from __future__ import annotations

from fake_chat_model import RecordingChatModel

from injection_guard.config import get_settings
from injection_guard.defense.safety_detector import SafetyDetector
from injection_guard.services.llm_server import ChatCompletionRequest


def test_output_detector_model_is_deterministic_without_length_limit() -> None:
    """检测模型：temperature=0（确定性），且不设生成长度（交给服务端）。"""
    detector = SafetyDetector()
    assert detector.model.temperature == 0.0
    assert detector.model.max_tokens is None


def test_output_detection_client_sends_no_max_tokens() -> None:
    detector = SafetyDetector()
    model = RecordingChatModel(replies=["safe"])
    detector.model = model  # type: ignore[assignment]
    detector.rules = [{"id": "S1", "critique_request": "x"}]

    assert detector.detect("user text", "answer text").is_safe is True
    # 检测调用不携带任何生成参数
    assert model.calls[0]["kwargs"] == {}


def test_chat_completion_request_accepts_max_completion_tokens() -> None:
    """LangChain 客户端发 max_completion_tokens：服务端两个字段名都认，现行字段优先。"""
    messages = [{"role": "user", "content": "hi"}]
    assert ChatCompletionRequest(messages=messages, max_tokens=3).effective_max_tokens() == 3
    assert ChatCompletionRequest(messages=messages, max_completion_tokens=5).effective_max_tokens() == 5
    both = ChatCompletionRequest(messages=messages, max_tokens=3, max_completion_tokens=5)
    assert both.effective_max_tokens() == 5


def test_output_detector_max_tokens_field_removed() -> None:
    """应用侧已删除该字段（避免与服务端 <>_API_SERVER_MAX_NEW_TOKENS 重复）。"""
    assert not hasattr(get_settings(), "output_detector_max_tokens")
