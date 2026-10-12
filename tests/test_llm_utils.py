"""``llm_utils`` 文本工具的单元测试（不依赖模型）。"""

from __future__ import annotations

from injection_guard.llm_utils import estimate_tokens, message_text


def test_message_text_handles_str_list_and_other():
    assert message_text("abc") == "abc"
    assert message_text([{"type": "text", "text": "a"}, "b", {"type": "image"}]) == "ab"
    assert message_text({"unexpected": 1}) == ""


def test_estimate_tokens_cjk_and_ascii():
    assert estimate_tokens("") == 0
    assert estimate_tokens("安全") == 2            # 中日韩字符 1 字 ≈ 1 token
    assert estimate_tokens("abcd") == 1            # 4 个 ASCII 字符 ≈ 1 token
    assert estimate_tokens("a") == 1               # 非空至少 1
    assert estimate_tokens("安全abcd") == 3        # 2 + 1
