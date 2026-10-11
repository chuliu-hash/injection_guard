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
    validate_defend_request,
)
from injection_guard.api.security import get_current_user
from injection_guard.services import connectivity


class _FakeSettings:
    default_sensitivity = "low"
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
        self.detect_kwargs = _kwargs
        if self._exc is not None:
            raise self._exc
        assert self._outcome is not None
        return self._outcome


def _client(service: _FakeService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_defense_service] = lambda: service
    # 鉴权已由 auth 测试覆盖，这里只测流水线：桩掉当前用户
    app.dependency_overrides[get_current_user] = lambda: object()
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
    assert body["default_sensitivity"] == "low"
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


def test_generation_params_reach_service():
    """路由把请求体的 temperature / max_tokens 透传给防御服务。"""
    svc = _FakeService(outcome=DefendOutcome(True, None, SAFE_INPUT, None))
    client = _client(svc)
    resp = client.post(
        "/api/prompt_defend",
        json={"user_input": "你好", "temperature": 0.3, "max_tokens": 64},
    )
    assert resp.status_code == 200
    assert svc.detect_kwargs["temperature"] == 0.3
    assert svc.detect_kwargs["max_tokens"] == 64


def test_sensitivity_reaches_service():
    """请求体 sensitivity 透传给防御服务；缺省为 None（后端按默认值处理）。"""
    svc = _FakeService(outcome=DefendOutcome(True, None, SAFE_INPUT, None))
    client = _client(svc)
    resp = client.post("/api/prompt_defend", json={"user_input": "你好", "sensitivity": "high"})
    assert resp.status_code == 200
    assert svc.detect_kwargs["sensitivity"] == "high"

    client.post("/api/prompt_defend", json={"user_input": "你好"})
    assert svc.detect_kwargs["sensitivity"] is None


def test_invalid_sensitivity_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/prompt_defend", json={"user_input": "hi", "sensitivity": "extreme"})
    assert resp.status_code == 400
    assert resp.json() == {"error": "sensitivity 必须是 low / medium / high 之一"}


def test_safe_skip_threshold_out_of_range_returns_400():
    client = _client(_FakeService())
    resp = client.post(
        "/api/prompt_defend", json={"user_input": "hi", "safe_skip_threshold": 1.5}
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "safe_skip_threshold 必须在 0~1 之间"}


def test_validate_rejects_non_bool_skip_flag():
    # 直接测校验函数（经 FastAPI/Pydantic 时非法布尔值会先被 422 拦下）
    ok, message = validate_defend_request(
        {"user_input": "hi", "skip_output_when_high_confidence": "yes"}
    )
    assert ok is False
    assert message == "skip_output_when_high_confidence 必须是布尔值"


def test_temperature_out_of_range_returns_400():
    client = _client(_FakeService())
    resp = client.post(
        "/api/prompt_defend", json={"user_input": "hi", "temperature": 2.5}
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "temperature 必须在 0~2 之间"}


def test_max_tokens_non_positive_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/prompt_defend", json={"user_input": "hi", "max_tokens": 0})
    assert resp.status_code == 400
    assert resp.json() == {"error": "max_tokens 必须 >= 1"}


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
    output = {"is_safe": True, "failed_rules": [], "total_rules": 12}
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
    assert body["output_detection"]["total_rules"] == 12
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


# ── POST /api/test_services（连通性测试）────────────────────────────────


def _stub_check_services(monkeypatch, recorder: Dict[str, Any]):
    """把 connectivity.check_services 换成桩：记录入参并返回固定报告。"""

    def fake(settings, services=None, timeout=None):
        recorder["settings"] = settings
        recorder["services"] = services
        recorder["timeout"] = timeout
        keys = list(services) if services else list(connectivity.SERVICE_KEYS)
        return {
            "ok": True,
            "timeout": connectivity.normalize_timeout(timeout),
            "services": {
                key: {
                    "ok": True,
                    "label": key,
                    "url": "http://x/v1",
                    "model": "m",
                    "latency_ms": 1.0,
                    "error": None,
                }
                for key in keys
            },
        }

    monkeypatch.setattr(connectivity, "check_services", fake)


def test_test_services_defaults_to_all(monkeypatch):
    recorder: Dict[str, Any] = {}
    _stub_check_services(monkeypatch, recorder)
    service = _FakeService()

    resp = _client(service).post("/api/test_services")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["timeout"] == connectivity.DEFAULT_TIMEOUT
    assert set(body["services"]) == set(connectivity.SERVICE_KEYS)
    assert body["services"]["chat"]["ok"] is True
    # 桩收到的是 DefenseService 的 settings（而非重新 get_settings）
    assert recorder["settings"] is service.settings
    assert recorder["services"] is None


def test_test_services_subset_and_timeout(monkeypatch):
    recorder: Dict[str, Any] = {}
    _stub_check_services(monkeypatch, recorder)

    resp = _client(_FakeService()).post(
        "/api/test_services", json={"services": ["chat"], "timeout": 2.5}
    )

    assert resp.status_code == 200
    assert set(resp.json()["services"]) == {"chat"}
    assert recorder["services"] == ["chat"]
    assert recorder["timeout"] == 2.5


def test_test_services_unknown_name_returns_400():
    resp = _client(_FakeService()).post("/api/test_services", json={"services": ["nope"]})
    assert resp.status_code == 400
    body = resp.json()
    assert "error" in body
    assert "nope" in body["error"]


def test_test_services_empty_list_returns_400():
    resp = _client(_FakeService()).post("/api/test_services", json={"services": []})
    assert resp.status_code == 400
    assert resp.json() == {"error": "services 不能为空列表"}


def test_test_services_timeout_out_of_range_returns_400():
    client = _client(_FakeService())
    resp = client.post("/api/test_services", json={"timeout": 99})
    assert resp.status_code == 400
    assert "timeout" in resp.json()["error"]

    resp = client.post("/api/test_services", json={"timeout": 0.1})
    assert resp.status_code == 400
