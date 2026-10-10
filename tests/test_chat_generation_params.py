"""聊天生成参数：请求级覆盖 > Settings 默认值。

``ChatClient`` 的 ``temperature`` / ``max_tokens`` 默认取自 ``Settings``（不再是
env 文件里的客户端参数），前端设置面板随请求体发送时按请求覆盖。这里不发起网络请求。
"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.config import get_settings
from injection_guard.services.chat_client import ChatClient


class _FakeCompletions:
    def __init__(self, captured: dict) -> None:
        self._captured = captured

    def create(self, **kwargs: object) -> object:
        self._captured.update(kwargs)
        if kwargs.get("stream"):
            return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="h"))])])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))]
        )


def _client() -> tuple[ChatClient, dict]:
    captured: dict = {}
    client = ChatClient()
    client.client = SimpleNamespace(  # type: ignore[assignment]
        chat=SimpleNamespace(completions=_FakeCompletions(captured))
    )
    return client, captured


def test_request_params_override_defaults() -> None:
    client, captured = _client()
    client.complete("你好", max_tokens=7, temperature=0.1)
    assert captured["max_tokens"] == 7
    assert captured["temperature"] == 0.1


def test_omitted_params_fall_back_to_settings() -> None:
    client, captured = _client()
    client.complete("你好")
    assert captured["max_tokens"] == get_settings().chat_max_tokens
    assert captured["temperature"] == get_settings().chat_temperature


def test_stream_forwards_request_params() -> None:
    client, captured = _client()
    list(client.stream("你好", max_tokens=11, temperature=0.9))
    assert captured["max_tokens"] == 11
    assert captured["temperature"] == 0.9
    assert captured["stream"] is True


def test_settings_keeps_chat_defaults() -> None:
    """客户端参数仍在 Settings 中保留默认值（只是不写入 env 文件）。"""
    settings = get_settings()
    assert settings.chat_max_tokens == 1024
    assert settings.chat_temperature == 0.7
