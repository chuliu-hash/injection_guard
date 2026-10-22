"""聊天生成参数：请求级覆盖 > Settings 默认值。

``ChatClient`` 基于 LangChain ``ChatOpenAI``：``temperature`` / ``max_tokens`` 默认取自
``Settings``，前端设置面板随请求体发送时按请求经 ``bind`` 覆盖。这里用假 ChatModel，
不发起网络请求。
"""

from __future__ import annotations

from fake_chat_model import RecordingChatModel
from langchain_core.messages import HumanMessage

from injection_guard.config import get_settings
from injection_guard.services.chat_client import ChatClient


def _client(reply: str = "hi") -> tuple[ChatClient, RecordingChatModel]:
    client = ChatClient()
    model = RecordingChatModel(replies=[reply])
    client.model = model  # type: ignore[assignment]
    return client, model


def test_request_params_override_defaults() -> None:
    client, model = _client()
    assert client.complete(
        "你好",
        max_tokens=7,
        temperature=0.1,
        top_p=0.8,
        frequency_penalty=0.2,
        presence_penalty=0.3,
        seed=42,
    ) == "hi"
    assert model.calls[0]["kwargs"] == {
        "temperature": 0.1,
        "max_tokens": 7,
        "top_p": 0.8,
        "frequency_penalty": 0.2,
        "presence_penalty": 0.3,
        "seed": 42,
    }


def test_omitted_params_fall_back_to_settings() -> None:
    client, model = _client()
    client.complete("你好")
    settings = get_settings()
    assert model.calls[0]["kwargs"] == {
        "temperature": settings.chat_temperature,
        "max_tokens": settings.chat_max_tokens,
        "top_p": settings.chat_top_p,
        "frequency_penalty": settings.chat_frequency_penalty,
        "presence_penalty": settings.chat_presence_penalty,
    }


def test_stream_forwards_request_params() -> None:
    client, model = _client("hello")
    assert list(client.stream("你好", max_tokens=11, temperature=0.9)) == ["hello"]
    assert model.calls[0]["kwargs"] == {
        "temperature": 0.9,
        "max_tokens": 11,
        "top_p": get_settings().chat_top_p,
        "frequency_penalty": get_settings().chat_frequency_penalty,
        "presence_penalty": get_settings().chat_presence_penalty,
    }


def test_prompt_is_rendered_as_single_user_message() -> None:
    """ChatPromptTemplate 负责渲染：prompt 内容按值插入（花括号不会被二次解析）。"""
    client, model = _client()
    client.complete("{花括号} 原样送达")
    messages = model.calls[0]["messages"]
    assert len(messages) == 1
    assert messages[0].type == "human"
    assert messages[0].content == "{花括号} 原样送达"


def test_wire_payload_uses_openai_max_completion_tokens() -> None:
    """langchain-openai 会把 max_tokens 写成官方字段 max_completion_tokens。

    推理服务（``llm_server.ChatCompletionRequest``）已同时接受两个字段名。
    """
    client = ChatClient()
    payload = client.model._get_request_payload([HumanMessage(content="hi")], max_tokens=7)  # noqa: SLF001
    assert payload["max_completion_tokens"] == 7
    assert "max_tokens" not in payload


def test_settings_keeps_chat_defaults() -> None:
    """客户端参数仍在 Settings 中保留默认值（只是不写入 env 文件）。"""
    settings = get_settings()
    assert settings.chat_max_tokens == 1024
    assert settings.chat_temperature == 0.7
    assert settings.chat_top_p == 0.9
