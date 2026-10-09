"""防御 API 的端到端测试（用桩 DefenseService，不加载真实模型）。

通过 ``app.dependency_overrides[get_defense_service]`` 注入假服务，
从而在不导入 torch / transformers 的前提下覆盖全部响应分支。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi.testclient import TestClient

from injection_guard.api.app import create_app
from injection_guard.api.dependencies import (
    DefendOutcome,
    DetectionError,
    ServiceBusyError,
    get_defense_service,
)


class _FakeSettings:
    fixed_sensitivity = "low"
    skip_output_when_high_confidence = True
    safe_skip_threshold = 0.5
    enable_rag = False


class _FakeService:
    """最小化实现 routes 所需的属性与 ``detect`` 方法。"""

    def __init__(self, outcome: Optional[DefendOutcome] = None, exc: Optional[Exception] = None):
        self.settings = _FakeSettings()
        self.input_max_concurrency = 8
        self.output_max_concurrency = 8
        self.chat_max_concurrency = 8
        self.input_gpu_ids = []
        self.replicas = 1
        self._outcome = outcome
        self._exc = exc

    @property
    def input_detector_devices(self):
        return []

    def detect(self, **_kwargs: Any) -> DefendOutcome:
        if self._exc is not None:
            raise self._exc
        assert self._outcome is not None
        return self._outcome


def _client(service: _FakeService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_defense_service] = lambda: service
    return TestClient(app)


SAFE_INPUT: Dict[str, Any] = {
    "is_safe": True,
    "risk_level": "无风险",
    "category": "正常",
    "confidence": 0.99,
    "p_safe": 0.99,
    "p_attack": 0.01,
    "decision_by": "bert_ensemble_only",
}

PAYLOAD = {"user_input": "你好", "model_response": "你好！"}


def test_health():
    client = _client(_FakeService(outcome=DefendOutcome(True, None, SAFE_INPUT, None)))
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["fixed_sensitivity"] == "low"
    assert body["chat_max_concurrency"] == 8


def test_missing_user_input_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/prompt_defend", json={"model_response": "你好"})
    assert resp.status_code == 400
    assert resp.json() == {"error": "缺少必填字段: user_input"}


def test_empty_user_input_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/prompt_defend", json={"user_input": "  "})
    assert resp.status_code == 400
    assert resp.json() == {"error": "user_input 不能为空"}


def test_empty_model_response_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/prompt_defend", json={"user_input": "hi", "model_response": "  "})
    assert resp.status_code == 400
    assert resp.json() == {"error": "model_response 不能为空"}


def test_input_blocked_stops_at_input_detection():
    blocked = {**SAFE_INPUT, "is_safe": False, "risk_level": "高风险", "category": "指令覆盖"}
    outcome = DefendOutcome(False, "input_detection", blocked, None)
    client = _client(_FakeService(outcome=outcome))

    resp = client.post("/api/prompt_defend", json=PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_safe"] is False
    assert body["stopped_at"] == "input_detection"
    assert body["input_detection"]["risk_level"] == "高风险"
    assert body["output_detection"] is None
    assert body["model_response"] is None


def test_high_confidence_skips_output_but_returns_response():
    outcome = DefendOutcome(True, "output_skipped", SAFE_INPUT, None, "生成的回复", None)
    client = _client(_FakeService(outcome=outcome))

    resp = client.post("/api/prompt_defend", json={"user_input": "你好"})
    body = resp.json()
    assert body["is_safe"] is True
    assert body["stopped_at"] == "output_skipped"
    assert body["model_response"] == "生成的回复"
    assert body["input_detection"]["p_safe"] == 0.99
    assert body["output_detection"] is None


def test_full_pipeline_returns_both_detections_and_rag():
    output = {"is_safe": True, "failed_rules": [], "total_rules": 5}
    rag = {
        "enabled": True,
        "final_prompt": "Safe Example 1 ...",
        "retrieved_contexts": [{"prompt": "p", "response": "r"}],
    }
    outcome = DefendOutcome(True, None, SAFE_INPUT, output, "模型回复", rag)
    client = _client(_FakeService(outcome=outcome))

    resp = client.post("/api/prompt_defend", json={"user_input": "你好", "use_rag": True})
    body = resp.json()
    assert body["stopped_at"] is None
    assert body["model_response"] == "模型回复"
    assert body["input_detection"]["is_safe"] is True
    assert body["output_detection"]["total_rules"] == 5
    assert body["output_detection"]["failed_rules"] == []
    assert body["rag"]["enabled"] is True
    assert body["rag"]["retrieved_contexts"][0]["response"] == "r"


def test_busy_returns_503():
    client = _client(_FakeService(exc=ServiceBusyError("输入检测服务繁忙")))
    resp = client.post("/api/prompt_defend", json=PAYLOAD)
    assert resp.status_code == 503
    assert resp.json() == {"error": "service_busy"}


def test_detection_error_returns_500():
    client = _client(_FakeService(exc=DetectionError("输入检测失败: boom")))
    resp = client.post("/api/prompt_defend", json=PAYLOAD)
    assert resp.status_code == 500
    assert resp.json() == {"error": "输入检测失败: boom"}


def test_non_json_body_returns_400():
    client = _client(_FakeService())
    resp = client.post(
        "/api/prompt_defend",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 400
    assert "error" in resp.json()
