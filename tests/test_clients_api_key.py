"""客户端 API key 兜底：本地留空 → 占位串；显式/配置提供 → 使用真 key。

openai SDK 不接受空 api_key（会抛 Missing credentials），故「留空」时用占位串，
本地服务忽略该 Authorization 头。这里不发起任何网络请求，只校验 key 选择逻辑。
"""

from __future__ import annotations

import pytest

from injection_guard.config import get_settings
from injection_guard.defense.safety_detector import SafetyDetector
from injection_guard.services.chat_client import ChatClient


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """lru_cache 会冻结首次读取；本机 .env 可能填了真 key，用例前后各清一次。"""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_chat_client_empty_key_falls_back_to_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    # os.environ 优先级最高，置空以屏蔽本机 .env 里的真 key，验证「留空 → 占位串」
    monkeypatch.setenv("CHAT_API_KEY", "")
    get_settings.cache_clear()
    assert ChatClient().api_key == "EMPTY"


def test_chat_client_explicit_key_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT_API_KEY", "")
    get_settings.cache_clear()
    assert ChatClient(api_key="sk-cloud-xxx").api_key == "sk-cloud-xxx"


def test_safety_detector_empty_key_falls_back_to_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OUTPUT_DETECTOR_API_KEY", "")
    get_settings.cache_clear()
    assert SafetyDetector().api_key == "EMPTY"


def test_safety_detector_explicit_key_wins() -> None:
    assert SafetyDetector(api_key="sk-cloud-xxx").api_key == "sk-cloud-xxx"


def test_settings_exposes_chat_api_key_field() -> None:
    # 字段存在且默认可为空（本地无鉴权）
    assert hasattr(get_settings(), "chat_api_key")
    assert hasattr(get_settings(), "output_detector_api_key")
