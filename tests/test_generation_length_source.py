"""生成长度的单一来源：输入/输出检测的客户端不发送 ``max_tokens``。

生成长度由推理服务按 ``--service`` 读取的 ``<服务>_API_SERVER_MAX_NEW_TOKENS`` 决定
（``llm_server._prepare_generation`` 取 ``min(request.max_tokens, 服务端上限)``）；
客户端发送 ``max_tokens`` 仅保留给聊天（``CHAT_MAX_TOKENS``）。
这里不发起网络请求，只断言客户端不再携带该参数。
"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.config import get_settings
from injection_guard.defense.safety_detector import SafetyDetector


def test_output_detection_client_sends_no_max_tokens() -> None:
    detector = SafetyDetector()
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="safe"))]
            )

    detector.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=_FakeCompletions())
    )

    rule_id, is_safe = detector._check_single_rule(
        "user text", "answer text", {"id": "r1", "critique_request": "x"}
    )

    assert (rule_id, is_safe) == ("r1", True)
    # 生成长度交给服务端；客户端不再传 max_tokens
    assert "max_tokens" not in captured
    # 检测仍需确定性，temperature 保留
    assert captured["temperature"] == 0.0


def test_output_detector_max_tokens_field_removed() -> None:
    """应用侧已删除该字段（避免与服务端 <>_API_SERVER_MAX_NEW_TOKENS 重复）。"""
    assert not hasattr(get_settings(), "output_detector_max_tokens")
