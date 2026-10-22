"""离线评测指标与用例加载（P1）。

评测与推理解耦：本模块只负责「用例 -> 预测 -> 指标」，真正的检测由调用方注入。
这样单测可以用桩，生产可以用 ``DefenseService`` 或外部 API 批量跑红队集。

指标口径：

- ``expected_safe=False`` 视为攻击 / 不安全样本（正类）；
- ``attack_success_rate`` = 漏报数 / 攻击样本数，越低越好；
- ``false_positive_rate`` = 误报数 / 安全样本数，越低越好。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EvalCase:
    """一条评测用例。"""

    id: str
    user_input: str
    expected_safe: bool
    category: str = "default"
    model_response: str = "这是一个用于评测的占位回复。"
    sensitivity: Optional[str] = None


def load_cases(path: str | Path) -> List[EvalCase]:
    """从 JSONL 读取用例；每行至少包含 ``user_input`` 与 ``expected_safe``。"""
    cases: List[EvalCase] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"第 {line_no} 行不是合法 JSON: {exc}") from exc
            if "user_input" not in payload or "expected_safe" not in payload:
                raise ValueError(f"第 {line_no} 行缺少 user_input / expected_safe")
            cases.append(
                EvalCase(
                    id=str(payload.get("id") or f"case-{line_no}"),
                    user_input=str(payload["user_input"]),
                    expected_safe=bool(payload["expected_safe"]),
                    category=str(payload.get("category") or "default"),
                    model_response=str(payload.get("model_response") or "这是一个用于评测的占位回复。"),
                    sensitivity=payload.get("sensitivity"),
                )
            )
    return cases


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def compute_metrics(rows: Iterable[Tuple[bool, bool, str]]) -> Dict[str, Any]:
    """根据 ``(expected_safe, predicted_safe, category)`` 计算总体与分类指标。"""
    overall = _confusion()
    by_category: Dict[str, Dict[str, Any]] = {}
    for expected_safe, predicted_safe, category in rows:
        _update(overall, expected_safe, predicted_safe)
        _update(by_category.setdefault(category, _confusion()), expected_safe, predicted_safe)

    return {
        **_finalize(overall),
        "categories": {name: _finalize(values) for name, values in sorted(by_category.items())},
    }


def _confusion() -> Dict[str, int]:
    return {"total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}


def _update(values: Dict[str, int], expected_safe: bool, predicted_safe: bool) -> None:
    values["total"] += 1
    if expected_safe and predicted_safe:
        values["tn"] += 1
    elif expected_safe and not predicted_safe:
        values["fp"] += 1
    elif not expected_safe and not predicted_safe:
        values["tp"] += 1
    else:
        values["fn"] += 1


def _finalize(values: Dict[str, int]) -> Dict[str, Any]:
    tp, tn, fp, fn = values["tp"], values["tn"], values["fp"], values["fn"]
    positives = tp + fn
    negatives = tn + fp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positives)
    return {
        **values,
        "accuracy": _ratio(tp + tn, values["total"]),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": _ratio(fp, negatives),
        "false_negative_rate": _ratio(fn, positives),
        "attack_success_rate": _ratio(fn, positives),
    }


def evaluate_cases(
    cases: Sequence[EvalCase],
    predict: Callable[[EvalCase], bool],
) -> Dict[str, Any]:
    """执行用例并返回指标；``predict`` 返回 ``predicted_safe``。"""
    rows = [(case.expected_safe, bool(predict(case)), case.category) for case in cases]
    return compute_metrics(rows)


def evaluate_with_service(service: Any, cases: Sequence[EvalCase]) -> Dict[str, Any]:
    """用 ``DefenseService.detect`` 跑评测（生产路径，需要模型可用）。"""

    def predict(case: EvalCase) -> bool:
        outcome = service.detect(
            user_input=case.user_input,
            model_response=case.model_response,
            sensitivity=case.sensitivity,
        )
        return bool(outcome.is_safe)

    return evaluate_cases(cases, predict)
