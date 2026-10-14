"""输出检测：单次调用 + 判定词解析 + 异常默认放行（纯逻辑，不联网、不加载模型）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest
from fake_chat_model import RecordingChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from injection_guard.defense.safety_detector import SafetyDetector, SafetyVerdict


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
        ('{"verdict": "unsafe", "categories": ["S5"]}', False, ["S5"]),
        ('{"verdict": "unsafe", "categories": ["s5", "S99", "S5"]}', False, ["S5"]),
        ('{"verdict": "SAFE", "categories": []}', True, []),
        ('{"verdict": "unsafe"}', False, []),
        ('```json\n{"verdict": "unsafe", "categories": ["S1"]}\n```', False, ["S1"]),
        ('{"verdict": "unsafe", "categories": "S9"}', False, ["S9"]),
    ],
)
def test_detect_parses_structured_json_verdict(
    content: str, expected_safe: bool, expected_failed: list[str]
) -> None:
    """结构化输出：JSON 一次解析到位（不需要修复重试），类别只认已加载的规则。"""
    detector, fake = _detector(content)
    result = detector.detect("用户输入", "模型回复")

    assert result.is_safe is expected_safe
    assert result.failed_rules == expected_failed
    assert len(fake.calls) == 1


def test_unparseable_reply_triggers_one_repair_retry() -> None:
    """模型没按格式回：再要一次严格 JSON，第二次解析成功即采用。"""
    detector = SafetyDetector.__new__(SafetyDetector)
    detector.model_name = "test-model"
    detector.rules = [{"id": "S5", "critique_request": "暴力犯罪。"}]
    model = RecordingChatModel(replies=["嗯……这个我不好说", '{"verdict": "unsafe", "categories": ["S5"]}'])
    detector.model = model  # type: ignore[assignment]

    result = detector.detect("用户输入", "模型回复")

    assert result.is_safe is False
    assert result.failed_rules == ["S5"]
    assert len(model.calls) == 2
    # 修复请求把上次的原始回复带上，并要求只输出 JSON
    assert model.calls[1]["messages"][-1].content == "嗯……这个我不好说"
    assert "JSON" in model.calls[1]["messages"][0].content


def test_repair_retry_still_unparseable_defaults_to_pass() -> None:
    """修复重试仍然解析不了：只重试一次，然后默认放行。"""
    detector = SafetyDetector.__new__(SafetyDetector)
    detector.model_name = "test-model"
    detector.rules = [{"id": "S5", "critique_request": "暴力犯罪。"}]
    model = RecordingChatModel(replies=["嗯……", "还是不想说"])
    detector.model = model  # type: ignore[assignment]

    result = detector.detect("用户输入", "模型回复")

    assert result.is_safe is True
    assert len(model.calls) == 2


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
    # 提示模型输出结构化 JSON（判定 + 命中类别）
    assert '"verdict"' in system_prompt and '"categories"' in system_prompt
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
    # 判定词仍是英文 safe / unsafe，且要求结构化 JSON 输出
    assert '"verdict"' in system_prompt and '"unsafe"' in system_prompt
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


# ── 结构化输出（with_structured_output + response_format）────────────────


class _StructuredModelStub:
    """最小模型桩：``with_structured_output`` 返回预设结果，``invoke`` 供修复重试用。"""

    def __init__(self, structured_result=None, builder_exc: Optional[Exception] = None, reply: str = ""):
        self.structured_result = structured_result
        self.builder_exc = builder_exc
        self.reply = reply
        self.bound: Optional[tuple] = None
        self.invocations: list = []
        self.repair_calls: list = []
        stub = self

        class _Runnable:
            def invoke(self, messages):
                stub.invocations.append(messages)
                return stub.structured_result

        self._runnable = _Runnable()

    def invoke(self, messages, **kwargs):
        self.repair_calls.append(messages)
        return AIMessage(content=self.reply)

    def with_structured_output(self, schema, *, method=None, include_raw=False):
        if self.builder_exc is not None:
            raise self.builder_exc
        self.bound = (schema, method, include_raw)
        return self._runnable


def _detector_with_stub(stub) -> SafetyDetector:
    detector = SafetyDetector.__new__(SafetyDetector)
    detector.model_name = "test-model"
    detector.rules = [{"id": "S5", "critique_request": "暴力犯罪。"}]
    detector.model = stub  # type: ignore[assignment]
    detector._structured_cache = None
    return detector


def test_structured_output_is_preferred() -> None:
    """结构化路径可用时：用 with_structured_output + json_object，一次调用拿到判定。"""
    verdict = SafetyVerdict(verdict="unsafe", categories=["S5"])
    stub = _StructuredModelStub(structured_result={"raw": AIMessage(content=verdict.model_dump_json()), "parsed": verdict, "parsing_error": None})
    result = _detector_with_stub(stub).detect("用户输入", "模型回复")

    assert (result.is_safe, result.failed_rules) == (False, ["S5"])
    schema, method, include_raw = stub.bound
    assert schema is SafetyVerdict and method == "json_mode" and include_raw is True
    assert len(stub.invocations) == 1  # 结构化调用一次
    assert stub.repair_calls == []  # 不需要修复重试


def test_structured_raw_safe_verdict() -> None:
    verdict = SafetyVerdict(verdict="safe", categories=[])
    stub = _StructuredModelStub(
        structured_result={"raw": AIMessage(content="{}"), "parsed": verdict, "parsing_error": None}
    )
    result = _detector_with_stub(stub).detect("用户输入", "模型回复")
    assert result.is_safe is True and result.failed_rules == []


def test_structured_parse_error_goes_to_repair_once() -> None:
    """结构化解析失败：拿原始回复做一次修复重试，成功即采用（不再重复主调用）。"""
    stub = _StructuredModelStub(
        structured_result={
            "raw": AIMessage(content="嗯……这个我不好说"),
            "parsed": None,
            "parsing_error": ValueError("invalid json"),
        },
        reply='{"verdict": "unsafe", "categories": ["S5"]}',
    )
    result = _detector_with_stub(stub).detect("用户输入", "模型回复")

    assert (result.is_safe, result.failed_rules) == (False, ["S5"])
    assert len(stub.invocations) == 1  # 主调用只发生一次
    assert len(stub.repair_calls) == 1  # 修复重试一次
    assert stub.repair_calls[0][-1].content == "嗯……这个我不好说"


@pytest.mark.parametrize("exc", [NotImplementedError("stub"), ValueError("response_format not supported")])
def test_structured_unavailable_falls_back_to_plain_call(exc: Exception) -> None:
    """模型桩或服务端不支持结构化参数时，回落普通调用 + JSON 解析。"""
    stub = _StructuredModelStub(builder_exc=exc, reply='{"verdict": "unsafe", "categories": ["S5"]}')
    result = _detector_with_stub(stub).detect("用户输入", "模型回复")

    assert (result.is_safe, result.failed_rules) == (False, ["S5"])
    assert stub.invocations == []  # 没走结构化
    assert len(stub.repair_calls) == 1  # 走普通调用


def test_structured_call_failure_falls_back_to_plain_call() -> None:
    """结构化调用抛异常（服务端拒收 / 网络错）时回落普通调用。"""
    stub = _StructuredModelStub(reply="unsafe")
    stub._runnable = _BoomRunnable()
    result = _detector_with_stub(stub).detect("用户输入", "模型回复")

    assert result.is_safe is False
    assert len(stub.repair_calls) == 1


class _BoomRunnable:
    """调用即失败的 Runnable（模拟服务端拒收 response_format）。"""

    def invoke(self, messages):
        raise RuntimeError("response_format not supported")


@pytest.mark.parametrize(
    "payload",
    [
        {"verdict": "UNSAFE", "categories": ["S5"]},   # 大小写
        {"verdict": " unsafe "},                        # 前后空格 + 缺 categories
        {"verdict": "unsafe", "categories": "S5"},      # 类别给了单值字符串
    ],
)
def test_safety_verdict_normalizes_model_output(payload: dict) -> None:
    """结构化校验前先归一，避免大小写 / 单值类别这类小毛病触发回落。"""
    verdict = SafetyVerdict.model_validate(payload)
    assert verdict.verdict == ("unsafe" if "unsafe" in str(payload["verdict"]).lower() else "safe")
    assert verdict.categories == ["S5"] if payload.get("categories") else verdict.categories == []


def test_safety_verdict_rejects_unknown_verdict() -> None:
    """判定词不在 safe / unsafe 里：校验失败（由调用方回落文本解析）。"""
    with pytest.raises(ValidationError):
        SafetyVerdict.model_validate({"verdict": "也许安全"})
