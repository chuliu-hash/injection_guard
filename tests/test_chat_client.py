"""请求级聊天客户端配置：transport 覆盖与生成参数绑定。"""

from __future__ import annotations

from types import SimpleNamespace

from injection_guard.api.dependencies import DefenseService
from injection_guard.services import chat_client as chat_client_module
from injection_guard.services.chat_client import ChatClient


class _FakeChatModel:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.bound = None

    def bind(self, **kwargs):
        self.bound = kwargs
        return self

    def invoke(self, _messages):
        return SimpleNamespace(content="ok")

    def stream(self, _messages):
        yield SimpleNamespace(content="o")
        yield SimpleNamespace(content="k")


def test_chat_client_binds_request_generation_params(monkeypatch) -> None:
    models: list[_FakeChatModel] = []

    def _factory(**kwargs):
        model = _FakeChatModel(**kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(chat_client_module, "ChatOpenAI", _factory)
    client = ChatClient(
        base_url="https://example.com/v1",
        model_name="model-x",
        api_key="secret",
    )

    assert client.complete(
        "hello",
        temperature=0.2,
        max_tokens=128,
        top_p=0.8,
        frequency_penalty=0.3,
        presence_penalty=0.4,
        seed=42,
    ) == "ok"

    model = models[0]
    assert model.init_kwargs["base_url"] == "https://example.com/v1"
    assert model.init_kwargs["model"] == "model-x"
    assert model.init_kwargs["api_key"] == "secret"
    assert model.bound == {
        "temperature": 0.2,
        "max_tokens": 128,
        "top_p": 0.8,
        "frequency_penalty": 0.3,
        "presence_penalty": 0.4,
        "seed": 42,
    }


def test_custom_base_url_does_not_inherit_server_api_key(monkeypatch) -> None:
    created = []

    class _Client:
        def __init__(self, **kwargs):
            created.append(kwargs)

    service = DefenseService.__new__(DefenseService)
    service.settings = SimpleNamespace(
        chat_api_url="http://localhost:8097/v1",
        chat_model="default-model",
        chat_api_key="server-secret",
    )
    service._chat_client = object()
    monkeypatch.setattr("injection_guard.services.chat_client.ChatClient", _Client)

    service._get_chat_client({"api_url": "https://example.com/v1"})

    assert created == [
        {
            "base_url": "https://example.com/v1",
            "model_name": None,
            "api_key": "EMPTY",
        }
    ]


def test_same_default_transport_reuses_server_client(monkeypatch) -> None:
    """前端回传默认 URL / 模型时，必须沿用已注入 .env Key 的常驻客户端。"""
    service = DefenseService.__new__(DefenseService)
    service.settings = SimpleNamespace(
        chat_api_url="https://api.deepseek.com/v1",
        chat_model="deepseek-flash",
        chat_api_key="server-secret",
    )
    cached = object()
    service._chat_client = cached

    client = service._get_chat_client(
        {"api_url": "https://api.deepseek.com/v1/", "model": "deepseek-flash"}
    )

    assert client is cached
