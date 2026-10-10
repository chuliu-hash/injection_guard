"""推理服务配置（InferenceSettings）按服务前缀读取 + 与应用配置（Settings）的分离。

不加载 torch：仅 import ``injection_guard.config``（pydantic / pydantic-settings）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from injection_guard.config import (
    InferenceSettings,
    get_inference_settings,
    get_settings,
    inference_env_name,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """lru_cache 会冻结首次读取，用例前后各清一次以保证隔离。"""
    get_settings.cache_clear()
    get_inference_settings.cache_clear()
    yield
    get_settings.cache_clear()
    get_inference_settings.cache_clear()


def test_inference_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # _env_file=None 显式忽略仓库 configs/services.env，验证纯默认值
    for key in (
        "API_SERVER_MAX_QUEUE_SIZE",
        "API_SERVER_MAX_NEW_TOKENS",
        "API_SERVER_MAX_INPUT_TOKENS",
        "API_SERVER_SPLIT_ACROSS_GPUS",
    ):
        monkeypatch.delenv(key, raising=False)
    settings = InferenceSettings(_env_file=None)
    assert settings.api_server_max_queue_size == 32
    # 默认不设生成上限（由请求 max_tokens 决定）
    assert settings.api_server_max_new_tokens is None
    assert settings.api_server_max_input_tokens == 4096
    assert settings.api_server_split_across_gpus is False


def test_detection_services_default_max_new_tokens(tmp_path: Path) -> None:
    """检测服务内置默认 16（无需写进 services.env）；聊天不设上限。"""
    empty = tmp_path / "empty.env"
    empty.write_text("", encoding="utf-8")
    assert InferenceSettings.for_service("input", env_file=empty).api_server_max_new_tokens == 16
    assert InferenceSettings.for_service("output", env_file=empty).api_server_max_new_tokens == 16
    assert InferenceSettings.for_service("chat", env_file=empty).api_server_max_new_tokens is None
    # 环境变量仍可覆盖（优先于内置默认）
    env_file = tmp_path / "override.env"
    env_file.write_text("INPUT_API_SERVER_MAX_NEW_TOKENS=64\n", encoding="utf-8")
    assert InferenceSettings.for_service("input", env_file=env_file).api_server_max_new_tokens == 64


def test_for_service_reads_prefixed_keys(tmp_path: Path) -> None:
    env_file = tmp_path / "services.env"
    env_file.write_text(
        "INPUT_API_SERVER_MAX_QUEUE_SIZE=7\n"
        "OUTPUT_API_SERVER_MAX_QUEUE_SIZE=9\n"
        "API_SERVER_MAX_QUEUE_SIZE=99\n",
        encoding="utf-8",
    )
    assert InferenceSettings.for_service("input", env_file=env_file).api_server_max_queue_size == 7
    assert InferenceSettings.for_service("output", env_file=env_file).api_server_max_queue_size == 9
    # 无前缀：只读 API_SERVER_*
    assert InferenceSettings.for_service(None, env_file=env_file).api_server_max_queue_size == 99


def test_env_var_overrides_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # os.environ 优先级高于 env 文件 —— 即 serve_llm.py 写入 <SVC>_API_SERVER_GPU_IDS 依赖的机制
    monkeypatch.setenv("INPUT_API_SERVER_MAX_QUEUE_SIZE", "7")
    get_inference_settings.cache_clear()
    assert get_inference_settings("input").api_server_max_queue_size == 7
    # 其它服务不受影响
    assert get_inference_settings("output").api_server_max_queue_size == 32


def test_get_inference_settings_caches_per_service() -> None:
    assert get_inference_settings("input") is get_inference_settings("input")
    assert get_inference_settings("input") is not get_inference_settings("output")


def test_inference_env_name_prefix() -> None:
    assert inference_env_name("input", "API_SERVER_GPU_IDS") == "INPUT_API_SERVER_GPU_IDS"
    assert inference_env_name(None, "API_SERVER_GPU_IDS") == "API_SERVER_GPU_IDS"


def test_api_server_fields_removed_from_app_settings() -> None:
    # 分离断言：推理字段不再出现在应用 Settings 上
    assert not hasattr(get_settings(), "api_server_max_queue_size")
    assert not hasattr(get_settings(), "api_server_max_new_tokens")
