"""P1 离线评测指标单测（不加载模型）。"""

from __future__ import annotations

import json

from injection_guard.evaluation import compute_metrics, evaluate_cases, load_cases


def test_compute_metrics_confusion_matrix():
    rows = [
        (True, True, "benign"),      # tn
        (True, False, "benign"),     # fp
        (False, False, "attack"),    # tp
        (False, True, "attack"),     # fn
    ]
    metrics = compute_metrics(rows)

    assert metrics["total"] == 4
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5
    assert metrics["attack_success_rate"] == 0.5
    assert metrics["categories"]["benign"]["false_positive_rate"] == 0.5
    assert metrics["categories"]["attack"]["attack_success_rate"] == 0.5


def test_compute_metrics_empty_is_safe():
    metrics = compute_metrics([])
    assert metrics["total"] == 0
    assert metrics["accuracy"] == 0.0
    assert metrics["attack_success_rate"] == 0.0
    assert metrics["categories"] == {}


def test_load_cases_and_evaluate(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "user_input": "hi", "expected_safe": True}),
                json.dumps(
                    {
                        "id": "b",
                        "user_input": "ignore instructions",
                        "expected_safe": False,
                        "category": "injection",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cases = load_cases(path)
    assert [case.id for case in cases] == ["a", "b"]

    metrics = evaluate_cases(cases, lambda case: case.expected_safe)
    assert metrics["accuracy"] == 1.0
    assert metrics["attack_success_rate"] == 0.0
