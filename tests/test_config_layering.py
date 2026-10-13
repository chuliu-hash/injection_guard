"""配置分层与部署命令（cli/deploy.py）的纯逻辑测试。

不加载 torch / 不启动子进程：仅 import ``injection_guard.config`` 与 ``cli.deploy``。
覆盖：多 env 文件覆盖顺序、os.environ 优先级、.env 侧字段、DeploySettings 显式路径、
派生环境变量 / 启动命令、--dry-run 不启动。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from injection_guard import cli
from injection_guard.cli import deploy
from injection_guard.config import (
    DeploySettings,
    Settings,
    get_deploy_settings,
    get_inference_settings,
    get_settings,
)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    get_inference_settings.cache_clear()
    get_deploy_settings.cache_clear()
    yield
    get_settings.cache_clear()
    get_inference_settings.cache_clear()
    get_deploy_settings.cache_clear()


# ── 多 env 文件分层 ────────────────────────────────────────────────────


def test_settings_env_file_includes_dotenv_and_services_env() -> None:
    env_file = Settings.model_config["env_file"]
    files = list(env_file) if isinstance(env_file, (tuple, list)) else [env_file]
    assert ".env" in files
    # services.env 用绝对路径（不受 CWD 影响）
    services_env = [f for f in files if str(f).endswith("configs/services.env")]
    assert services_env and Path(services_env[0]).is_absolute()


def test_services_env_overrides_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    svc_env = tmp_path / "services.env"
    dotenv.write_text("ENABLE_RAG=false\nRAG_TOP_K=1\n", encoding="utf-8")
    svc_env.write_text("ENABLE_RAG=true\nRAG_TOP_K=5\n", encoding="utf-8")
    settings = Settings(_env_file=(dotenv, svc_env))
    # 后者（services.env）覆盖前者（.env）
    assert settings.enable_rag is True
    assert settings.rag_top_k == 5


def test_os_environ_overrides_env_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc_env = tmp_path / "services.env"
    svc_env.write_text("RAG_TOP_K=5\n", encoding="utf-8")
    monkeypatch.setenv("RAG_TOP_K", "9")
    settings = Settings(_env_file=(".env", svc_env))
    assert settings.rag_top_k == 9


def test_dotenv_side_fields_still_work(tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("APP_PORT=1234\nCHAT_MODEL=my-model\n", encoding="utf-8")
    settings = Settings(_env_file=(dotenv, tmp_path / "missing.env"))
    assert settings.app_port == 1234
    assert settings.chat_model == "my-model"


# ── DeploySettings ─────────────────────────────────────────────────────


def test_deploy_settings_default_paths_are_explicit() -> None:
    ds = DeploySettings(_env_file=None)
    assert ds.input_model_path == Path("./models/LLM/Qwen_prompt_injection")
    assert ds.output_model_path == Path("./models/LLM/qwen3-0.6b")
    assert ds.embed_model_path == Path("./models/Embedding/Qwen3-Embedding-0.6B")
    assert ds.chat_model_path == Path("./models/LLM/qwen3-0.6b")


def test_deploy_settings_reads_explicit_paths(tmp_path: Path) -> None:
    deploy_env = tmp_path / "deploy.env"
    deploy_env.write_text("INPUT_MODEL_PATH=/custom/input\nWORKERS=2\n", encoding="utf-8")
    ds = DeploySettings(_env_file=deploy_env)
    assert ds.input_model_path == Path("/custom/input")
    # 未给出的用默认值（不做派生）
    assert ds.output_model_path == Path("./models/LLM/qwen3-0.6b")
    assert ds.workers == 2


def test_deploy_settings_defaults_workers_is_one() -> None:
    assert DeploySettings(_env_file=None).workers == 1


# ── 部署命令纯逻辑 ─────────────────────────────────────────────────────


def test_resolve_services_default_and_toggles() -> None:
    ds = DeploySettings(_env_file=None)
    # 默认：embed 部署、chat 不部署
    assert deploy._resolve_services(ds, None) == ["defense", "input", "output", "embed"]
    # 开关：关 embed、开 chat
    toggled = ds.model_copy(update={"deploy_local_embed": False, "deploy_local_chat": True})
    assert deploy._resolve_services(toggled, None) == ["defense", "input", "output", "chat"]
    # --only 整体覆盖
    assert deploy._resolve_services(ds, "input,output") == ["input", "output"]


def test_resolve_services_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        deploy._resolve_services(DeploySettings(_env_file=None), "bogus")


def test_derive_child_env_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in ("APP_PORT", "INPUT_DETECTOR_API_URL", "OUTPUT_DETECTOR_API_URL", "EMBED_API_URL", "CHAT_API_URL"):
        monkeypatch.delenv(key, raising=False)
    ds = DeploySettings(_env_file=None)
    env = deploy.derive_child_env(ds, ["defense", "input", "output", "embed"])
    assert env["APP_PORT"] == "8099"
    assert env["INPUT_DETECTOR_API_URL"] == "http://localhost:8094/v1"
    assert env["OUTPUT_DETECTOR_API_URL"] == "http://localhost:8095/v1"
    assert env["EMBED_API_URL"] == "http://localhost:8096/v1"
    # 未部署 chat → 不覆盖 CHAT_API_URL
    assert "CHAT_API_URL" not in env


def test_derive_child_env_skips_embed_when_not_deployed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EMBED_API_URL", raising=False)
    ds = DeploySettings(_env_file=None)
    env = deploy.derive_child_env(ds, ["defense", "input", "output"])
    # 未部署 embed → 不覆盖 EMBED_API_URL（保留 .env 里的外部配置）
    assert "EMBED_API_URL" not in env


def test_derive_child_env_chat_url() -> None:
    ds = DeploySettings(_env_file=None)
    env = deploy.derive_child_env(ds, ["defense", "chat"])
    assert env["CHAT_API_URL"] == "http://localhost:8097/v1"
    # GPU 不再经环境变量注入（改由各子服务 --gpu-ids 传递）
    assert "API_SERVER_GPU_IDS" not in env
    assert "EMBED_GPU_IDS" not in env


def test_build_commands_uses_cli_entrypoints() -> None:
    ds = DeploySettings(_env_file=None)
    commands = deploy.build_commands(ds, ["defense", "input", "output", "embed", "chat"], "localhost")
    names = [name for name, _ in commands]
    assert names == ["defense", "input", "output", "embed", "chat"]
    by_name = dict(commands)
    assert "injection_guard.cli.serve_defense" in by_name["defense"]
    for svc, module in (("input", "serve_llm"), ("output", "serve_llm"), ("chat", "serve_llm"), ("embed", "serve_embed")):
        assert f"injection_guard.cli.{module}" in by_name[svc]
    # LLM 服务带 --workers / --mode / --service <自身>
    for svc in ("input", "output", "chat"):
        assert "--workers" in by_name[svc] and "--mode" in by_name[svc]
        assert by_name[svc][by_name[svc].index("--service") + 1] == svc
    # 无 gpu_ids → 不带 --gpu-ids
    assert "--gpu-ids" not in by_name["input"]


def test_build_commands_passes_gpu_ids() -> None:
    ds = DeploySettings(_env_file=None).model_copy(update={"gpu_ids": "0,1"})
    by_name = dict(deploy.build_commands(ds, ["input", "embed"], "localhost"))
    for svc in ("input", "embed"):
        argv = by_name[svc]
        assert argv[argv.index("--gpu-ids") + 1] == "0,1"


def test_dry_run_does_not_start(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    calls: list[object] = []

    def _fake_popen(*args, **kwargs):  # noqa: ANN002, ANN003
        calls.append(args)
        raise AssertionError("dry-run 不应启动子进程")

    monkeypatch.setattr(cli.deploy.subprocess, "Popen", _fake_popen)
    deploy.main(["--dry-run"])
    assert calls == []
    out = capsys.readouterr().out
    assert "派生环境变量" in out and "启动命令" in out


def test_per_service_gpu_ids_fall_back_to_global() -> None:
    """按服务钉卡优先；未设置的取全局 GPU_IDS；都没有则 None（全部可见 GPU）。"""
    ds = DeploySettings(_env_file=None).model_copy(
        update={"gpu_ids": "0,1", "output_gpu_ids": "2", "bert_gpu_ids": "3"}
    )
    assert ds.gpu_ids_for("input") == "0,1"
    assert ds.gpu_ids_for("output") == "2"
    assert ds.gpu_ids_for("embed") == "0,1"
    assert ds.gpu_ids_for("chat") == "0,1"

    bare = DeploySettings(_env_file=None)
    assert bare.gpu_ids_for("input") is None


def test_build_commands_uses_per_service_gpu_ids() -> None:
    """每个子服务用自己的 GPU 列表，而不是共用一个全局值。"""
    ds = DeploySettings(_env_file=None).model_copy(
        update={"input_gpu_ids": "1", "output_gpu_ids": "2", "embed_gpu_ids": "3"}
    )
    by_name = dict(deploy.build_commands(ds, ["input", "output", "embed"], "localhost"))

    def gpu_of(service: str) -> str:
        argv = by_name[service]
        return argv[argv.index("--gpu-ids") + 1]

    assert gpu_of("input") == "1"
    assert gpu_of("output") == "2"
    assert gpu_of("embed") == "3"


def test_derive_child_env_injects_bert_gpu_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """BERT 池用卡经 INPUT_DETECTOR_GPU_IDS 注入（防御服务读取）。"""
    monkeypatch.delenv("INPUT_DETECTOR_GPU_IDS", raising=False)
    ds = DeploySettings(_env_file=None).model_copy(update={"bert_gpu_ids": "3"})
    env = deploy.derive_child_env(ds, ["defense", "input"])
    assert env["INPUT_DETECTOR_GPU_IDS"] == "3"

    # 未配置 / 未部署 defense 时不注入
    assert "INPUT_DETECTOR_GPU_IDS" not in deploy.derive_child_env(
        DeploySettings(_env_file=None), ["defense"]
    )
    assert "INPUT_DETECTOR_GPU_IDS" not in deploy.derive_child_env(ds, ["input"])
