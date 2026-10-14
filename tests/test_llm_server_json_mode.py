"""本地推理服务对 OpenAI ``response_format`` 的等价支持。

服务端不做 logits 级约束解码（不引入 xgrammar / lm-format-enforcer 等新依赖），
而是「注入 JSON 指令 -> 生成 -> 校验 -> 不合法则重试一次」；这里只测纯函数、
指令注入位置与重试分支，不加载真实模型、不访问 GPU。
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from injection_guard.services import llm_server


def _request(**kwargs) -> llm_server.ChatCompletionRequest:
    return llm_server.ChatCompletionRequest(
        messages=[{"role": "user", "content": "hi"}], **kwargs
    )


def test_request_accepts_response_format() -> None:
    """langchain 的 with_structured_output 会发 response_format，服务端必须收得下。"""
    request = _request(response_format={"type": "json_object"})
    assert request.response_format == {"type": "json_object"}
    assert _request().response_format is None


def test_json_instruction_variants() -> None:
    assert llm_server._json_instruction(None) is None
    assert llm_server._json_instruction({}) is None
    assert llm_server._json_instruction({"type": "text"}) is None

    plain = llm_server._json_instruction({"type": "json_object"})
    assert plain is not None and "JSON" in plain

    with_schema = llm_server._json_instruction(
        {
            "type": "json_schema",
            "json_schema": {
                "name": "verdict",
                "schema": {"type": "object", "properties": {"verdict": {"type": "string"}}},
            },
        }
    )
    assert with_schema is not None and '"verdict"' in with_schema

    strict = llm_server._json_instruction({"type": "json_object"}, strict=True)
    assert strict is not None and "上一次输出不是合法 JSON" in strict


def test_json_text_tolerates_code_fence_and_chatter() -> None:
    assert llm_server._json_text('{"a": 1}') == '{"a": 1}'
    assert llm_server._json_text('好的：\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert llm_server._json_text('[{"a": 1}]') == '[{"a": 1}]'
    assert llm_server._json_text("没有 JSON") is None


def test_is_valid_json() -> None:
    assert llm_server._is_valid_json('{"verdict": "safe", "categories": []}') is True
    assert llm_server._is_valid_json('```json\n{"a": [1, 2]}\n```') is True
    assert llm_server._is_valid_json("安全") is False
    assert llm_server._is_valid_json("{不是 JSON}") is False


def test_messages_with_json_instruction_appends_at_tail() -> None:
    """指令追加为末条 user 消息（不污染 chat template 的角色标记）。"""
    original = _request().messages
    assert llm_server._messages_with_json_instruction(_request()) == original

    messages = llm_server._messages_with_json_instruction(_request(response_format={"type": "json_object"}))
    assert len(messages) == 2
    assert messages[-1].role == "user" and "JSON" in messages[-1].content


def test_prepare_generation_injects_instruction(monkeypatch) -> None:
    captured: dict = {}

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = list(messages)
            return "prompt"

        def __call__(self, prompt, **kwargs):
            return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))

    class FakeModel:
        pass

    monkeypatch.setattr(llm_server, "_tokenizer", FakeTokenizer())
    monkeypatch.setattr(llm_server, "_model", FakeModel())
    monkeypatch.setattr(
        llm_server,
        "_inference_settings",
        lambda: SimpleNamespace(api_server_max_input_tokens=1024, api_server_max_new_tokens=None),
    )

    llm_server._prepare_generation(_request(response_format={"type": "json_object"}))
    assert len(captured["messages"]) == 2
    assert "JSON" in captured["messages"][-1].content

    llm_server._prepare_generation(_request())
    assert len(captured["messages"]) == 1


def test_sync_retries_once_then_returns(monkeypatch) -> None:
    """首次输出不是合法 JSON -> 追加严格指令重试一次。"""
    calls: list = []

    def fake_run_generation(request, messages=None):
        calls.append(messages)
        text = "好的，这是 JSON：\n{\"verdict\": \"safe\"}" if len(calls) > 1 else "我认为是安全的"
        return None, text, [1, 2], 1, None

    monkeypatch.setattr(llm_server, "_run_generation", fake_run_generation)
    monkeypatch.setattr(llm_server, "_maybe_empty_cuda_cache", lambda *a, **k: None)

    response = llm_server._chat_completion_sync(_request(response_format={"type": "json_object"}))

    # 服务端原样返回模型文本（客户端解析器负责去掉前后寒暄），关键是不再重试
    assert '"verdict": "safe"' in response.choices[0].message.content
    assert len(calls) == 2
    assert calls[0] is None  # 首次用默认（尾部已注入非严格指令）
    assert "上一次输出不是合法 JSON" in calls[1][-1].content


def test_sync_does_not_retry_when_already_json(monkeypatch) -> None:
    calls: list = []

    def fake_run_generation(request, messages=None):
        calls.append(messages)
        return None, '{"verdict": "safe"}', [1, 2], 1, None

    monkeypatch.setattr(llm_server, "_run_generation", fake_run_generation)
    monkeypatch.setattr(llm_server, "_maybe_empty_cuda_cache", lambda *a, **k: None)

    response = llm_server._chat_completion_sync(_request(response_format={"type": "json_object"}))
    assert response.choices[0].message.content == '{"verdict": "safe"}'
    assert len(calls) == 1


def test_sync_without_response_format_never_retries(monkeypatch) -> None:
    calls: list = []

    def fake_run_generation(request, messages=None):
        calls.append(messages)
        return None, "随便一段自然语言", [1, 2], 1, None

    monkeypatch.setattr(llm_server, "_run_generation", fake_run_generation)
    monkeypatch.setattr(llm_server, "_maybe_empty_cuda_cache", lambda *a, **k: None)

    response = llm_server._chat_completion_sync(_request())
    assert response.choices[0].message.content == "随便一段自然语言"
    assert len(calls) == 1
