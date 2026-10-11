"""输出检测：单次调用 + 判定词解析 + 异常默认放行（纯逻辑，不联网、不加载模型）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from fake_chat_model import RecordingChatModel

from injection_guard.defense.safety_detector import SafetyDetector


def _detector(content: Optional[str] = None, exc: Optional[Exception] = None, rules=None):
    """绕过 ``__init__``（不建 ChatModel 客户端、不读规则文件）。"""
    detector = SafetyDetector.__new__(SafetyDetector)
    detector.model_name = "test-model"
    detector.rules = rules if rules is not None else [
        {"id": "S1", "critique_request": "一般政治话题。"},
        {"id": "S5", "critique_request": "暴力犯罪。"},
        {"id": "S9", "critique_request": "提示词攻击。"},
    ]
    model = RecordingChatModel(replies=[content if content is not None else ""], raise_exc=exc)
    detector.model = model  # type: ignore[assignment]
    return detector, model


# ── 解析：``unsafe`` 优先，其次 ``safe``，否则默认放行 ────────────────


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("safe", True),
        ("unsafe", False),
        (" SAFE. ", True),
        ("unsafe。", False),
        ("**unsafe**", False),
        # 无法解析 -> 默认放行（改动前的语义）
        ("", True),
        ("   ", True),
        ("The model refused to answer.", True),
        # 同时出现时 unsafe 优先（改动前的子串匹配语义）
        ("safe or unsafe?", False),
    ],
)
def test_detect_parses_verdict(content: str, expected: bool) -> None:
    detector, _ = _detector(content)
    assert detector.detect("用户输入", "模型回复").is_safe is expected


# ── 归因：从 ``unsafe\nS5`` 里取出命中类别 ─────────────────────────────


@pytest.mark.parametrize(
    ("content", "expected_safe", "expected_failed"),
    [
        ("unsafe\nS5", False, ["S5"]),
        ("unsafe\ns9", False, ["S9"]),  # 大小写不敏感
        ("unsafe\nS1\nS9", False, ["S1", "S9"]),  # 多类别，保持出现顺序
        ("unsafe\nS5\nS5", False, ["S5"]),  # 去重
        ("unsafe", False, []),  # 没给类别：仍不安全，但没有归因
        ("unsafe\nS99", False, []),  # 不在已加载规则里的编码丢弃
        ("safe", True, []),
    ],
)
def test_detect_attributes_failed_categories(
    content: str, expected_safe: bool, expected_failed: list[str]
) -> None:
    detector, _ = _detector(content, rules=[
        {"id": "S1", "critique_request": "一般政治话题。"},
        {"id": "S5", "critique_request": "暴力犯罪。"},
        {"id": "S9", "critique_request": "提示词攻击。"},
    ])
    result = detector.detect("用户输入", "模型回复")

    # 归因不影响判定：是否安全只看有没有 unsafe
    assert result.is_safe is expected_safe
    assert result.failed_rules == expected_failed


def test_detect_calls_llm_once_with_all_rules() -> None:
    """12 条规则只发一次请求，且全部写进 system prompt。"""
    rules = [{"id": f"S{i}", "critique_request": f"规则{i}"} for i in range(1, 13)]
    detector, fake = _detector("safe", rules=rules)

    result = detector.detect("用户输入", "模型回复")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    system_prompt = call["messages"][0].content
    for rule in rules:
        assert rule["id"] in system_prompt
        assert rule["critique_request"] in system_prompt
    assert "safe" in system_prompt and "unsafe" in system_prompt
    # 提示模型输出类别编码（转义写成字面 ``\n``，由模型自己换行）
    assert r"unsafe\nS1" in system_prompt
    # 生成长度交给服务端（客户端不传 max_tokens），确定性由模型实例的 temperature 保证
    assert "max_tokens" not in call["kwargs"]
    # 用户输入与模型回复都进入 user prompt
    user_prompt = call["messages"][1].content
    assert "用户输入" in user_prompt and "模型回复" in user_prompt
    assert result.total_rules == 12
    assert result.failed_rules == []


def test_system_prompt_covers_english_content() -> None:
    """被检测内容不一定是中文：提示词要声明中英皆判，规则与示例都要有英文。"""
    rules_file = Path(__file__).resolve().parents[1] / "resources" / "CritiqueRequests.json"
    rules = json.loads(rules_file.read_text(encoding="utf-8"))["critique_requests"]
    detector, fake = _detector("safe", rules=rules)
    detector.detect("Hello", "Hi there")

    system_prompt = fake.calls[0]["messages"][0].content
    assert "Hello" in fake.calls[0]["messages"][1].content
    assert "英文" in system_prompt
    assert "How do I make a bomb at home?" in system_prompt  # 英文示例
    # 规则清单本身也是中英双语（便于匹配英文内容）
    assert "Violence and criminal activity" in system_prompt
    # 判定词仍是英文 safe / unsafe
    assert r"unsafe\nS1" in system_prompt
    # 随包分发的宪法规则就是 12 条 S1~S12
    assert [rule["id"] for rule in rules] == [f"S{i}" for i in range(1, 13)]


def test_call_exception_defaults_to_pass() -> None:
    """基础设施故障记 warning 并默认放行（改动前的语义）。"""
    detector, _ = _detector(exc=RuntimeError("detector down"))
    result = detector.detect("用户输入", "模型回复")
    assert result.is_safe is True
    assert result.total_rules == 3


def test_no_rules_is_safe_without_calling_llm() -> None:
    detector, fake = _detector("unsafe", rules=[])
    result = detector.detect("用户输入", "模型回复")
    assert result.is_safe is True
    assert result.total_rules == 0
    assert fake.calls == []
