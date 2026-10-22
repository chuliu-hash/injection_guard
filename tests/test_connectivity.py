"""下游服务连通性探测的单元测试（不连真实服务）。

``_invoke`` / ``probe`` 被 monkeypatch 掉，故不产生网络请求，也不依赖 GPU。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from injection_guard.services import connectivity


def _fake_settings() -> SimpleNamespace:
    """只含 resolve_targets 会读取的字段。"""
    return SimpleNamespace(
        input_detector_api_url="http://in:1/v1",
        input_detector_api_model="in-model",
        output_detector_api_url="http://out:2/v1",
        output_detector_model="out-model",
        output_detector_api_key="out-key",
        chat_api_url="http://chat:3/v1",
        chat_model="chat-model",
        chat_api_key=None,
        embed_api_url="http://embed:4/v1",
        embed_model_name="embed-model",
    )


def test_resolve_targets_covers_all_services():
    targets = connectivity.resolve_targets(_fake_settings())
    assert list(targets) == list(connectivity.SERVICE_KEYS)

    assert targets["input"].base_url == "http://in:1/v1"
    assert targets["input"].model == "in-model"
    assert targets["input"].kind == "chat"
    assert targets["output"].model == "out-model"
    assert targets["rag"].kind == "embedding"
    assert targets["rag"].model == "embed-model"
    assert targets["chat"].base_url == "http://chat:3/v1"


def test_resolve_targets_falls_back_to_placeholder_key():
    """有 key 用真 key；没有的（input / rag / 未配置的 chat）用占位串。"""
    targets = connectivity.resolve_targets(_fake_settings())
    assert targets["output"].api_key == "out-key"
    assert targets["input"].api_key == connectivity._PLACEHOLDER_KEY
    assert targets["rag"].api_key == connectivity._PLACEHOLDER_KEY
    assert targets["chat"].api_key == connectivity._PLACEHOLDER_KEY


def test_resolve_targets_uses_request_chat_config():
    targets = connectivity.resolve_targets(
        _fake_settings(),
        chat_options={
            "api_url": "https://example.com/v1",
            "model": "model-x",
            "api_key": "secret",
        },
    )

    assert targets["chat"].base_url == "https://example.com/v1"
    assert targets["chat"].model == "model-x"
    assert targets["chat"].api_key == "secret"


def test_custom_chat_url_does_not_inherit_server_key():
    settings = _fake_settings()
    settings.chat_api_key = "server-secret"
    targets = connectivity.resolve_targets(
        settings, chat_options={"api_url": "https://example.com/v1"}
    )

    assert targets["chat"].api_key == connectivity._PLACEHOLDER_KEY


def test_same_chat_url_inherits_server_key():
    """前端回传服务端默认 URL 时，不应误判为自定义地址而丢弃 .env 中的 Key。"""
    settings = _fake_settings()
    settings.chat_api_key = "server-secret"
    targets = connectivity.resolve_targets(
        settings,
        chat_options={"api_url": settings.chat_api_url, "api_key": ""},
    )

    assert targets["chat"].api_key == "server-secret"


def test_same_chat_url_with_trailing_slash_inherits_server_key():
    settings = _fake_settings()
    settings.chat_api_key = "server-secret"
    targets = connectivity.resolve_targets(
        settings,
        chat_options={"api_url": settings.chat_api_url + "/"},
    )

    assert targets["chat"].api_key == "server-secret"


@pytest.mark.parametrize(
    ("given", "expected"),
    [(None, connectivity.DEFAULT_TIMEOUT), (0.1, connectivity.MIN_TIMEOUT), (999, connectivity.MAX_TIMEOUT), (3.5, 3.5)],
)
def test_normalize_timeout_clamps(given, expected):
    assert connectivity.normalize_timeout(given) == expected


def test_probe_success(monkeypatch):
    calls = []

    def fake_invoke(target, timeout):
        calls.append((target.key, timeout))

    monkeypatch.setattr(connectivity, "_invoke", fake_invoke)
    target = connectivity.resolve_targets(_fake_settings())["chat"]

    result = connectivity.probe(target, timeout=2.0)

    assert calls == [("chat", 2.0)]
    assert result["ok"] is True
    assert result["error"] is None
    assert result["model"] == "chat-model"
    assert result["url"] == "http://chat:3/v1"
    assert isinstance(result["latency_ms"], float)


def test_probe_failure_is_returned_not_raised(monkeypatch):
    def boom(target, timeout):
        raise ConnectionError("Connection refused")

    monkeypatch.setattr(connectivity, "_invoke", boom)
    target = connectivity.resolve_targets(_fake_settings())["rag"]

    result = connectivity.probe(target)

    assert result["ok"] is False
    assert "ConnectionError" in result["error"]
    assert "Connection refused" in result["error"]
    assert result["label"] == "RAG(Embedding)"


def test_check_services_runs_only_selected(monkeypatch):
    seen = []

    def fake_probe(target, timeout):
        seen.append(target.key)
        return {"ok": True, "label": target.label, "url": target.base_url, "model": target.model,
                "latency_ms": 1.0, "error": None}

    monkeypatch.setattr(connectivity, "probe", fake_probe)
    report = connectivity.check_services(_fake_settings(), services=["chat", "input"], timeout=1.0)

    assert sorted(seen) == ["chat", "input"]
    assert set(report["services"]) == {"input", "chat"}
    assert report["ok"] is True
    assert report["timeout"] == 1.0


def test_check_services_defaults_to_all(monkeypatch):
    monkeypatch.setattr(
        connectivity, "probe",
        lambda target, timeout: {"ok": True, "label": target.label, "url": target.base_url,
                                 "model": target.model, "latency_ms": 0.5, "error": None},
    )
    report = connectivity.check_services(_fake_settings())
    assert set(report["services"]) == set(connectivity.SERVICE_KEYS)


def test_check_services_one_failure_does_not_affect_others(monkeypatch):
    def fake_probe(target, timeout):
        ok = target.key != "rag"
        return {"ok": ok, "label": target.label, "url": target.base_url, "model": target.model,
                "latency_ms": 1.0, "error": None if ok else "boom"}

    monkeypatch.setattr(connectivity, "probe", fake_probe)
    report = connectivity.check_services(_fake_settings())

    assert report["ok"] is False  # 有任一失败则整体不通过
    assert report["services"]["chat"]["ok"] is True
    assert report["services"]["rag"]["error"] == "boom"


def test_check_services_ignores_unknown_names(monkeypatch):
    monkeypatch.setattr(
        connectivity, "probe",
        lambda target, timeout: {"ok": True, "label": target.label, "url": target.base_url,
                                 "model": target.model, "latency_ms": 0.5, "error": None},
    )
    report = connectivity.check_services(_fake_settings(), services=["nope"])
    assert report["services"] == {}
    assert report["ok"] is True  # 空集合的 all() 为 True；路由层负责对空/未知入参返回 400
