"""一键部署本地服务（替代旧 scripts/deploy_model.sh）。

从 ``configs/deploy.env``（``DeploySettings``）读取模型路径、端口、workers 等，
按端口派生并注入各子服务的连接 URL 环境变量，然后以子进程方式拉起：

- 防御 API（``serve_defense``，端口 ``DEFENSE_PORT``）
- 输入检测 LLM（``serve_llm``，端口 ``INPUT_PORT``）
- 输出检测 LLM（``serve_llm``，端口 ``OUTPUT_PORT``）
- Embedding（``serve_embed``，端口 ``EMBED_PORT``）——仅当 ``DEPLOY_LOCAL_EMBED=true``（默认）
- 聊天 LLM（``serve_llm``，端口 ``CHAT_PORT``）——仅当 ``DEPLOY_LOCAL_CHAT=true``（默认 false）

用法::

    injection-guard-deploy                 # 部署默认服务集
    injection-guard-deploy --dry-run       # 只打印派生环境变量与命令，不启动
    injection-guard-deploy --only input,output
    injection-guard-deploy --host 0.0.0.0 --url-host 10.0.0.5

优先级：``os.environ`` > ``configs/deploy.env`` > 默认值（见 ``config.DeploySettings``）。
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..config import DeploySettings, get_deploy_settings, get_settings
from ..logging_config import setup_logging

logger = logging.getLogger(__name__)

# 规范顺序；--only 未给时按此顺序部署默认集
ALL_SERVICES = ("defense", "input", "output", "embed", "chat")


def _resolve_services(ds: DeploySettings, only: str | None) -> list[str]:
    """确定要部署的服务列表：``--only`` 优先，否则按开关组装默认集。

    默认：``defense`` + ``input`` + ``output`` 恒部署；``embed`` 取决于
    ``DEPLOY_LOCAL_EMBED``；``chat`` 取决于 ``DEPLOY_LOCAL_CHAT``。
    """
    if only:
        services = [s.strip() for s in only.split(",") if s.strip()]
        unknown = [s for s in services if s not in ALL_SERVICES]
        if unknown:
            raise SystemExit(f"未知服务：{', '.join(unknown)}（可选：{', '.join(ALL_SERVICES)}）")
        return services
    services = ["defense", "input", "output"]
    if ds.deploy_local_embed:
        services.append("embed")
    if ds.deploy_local_chat:
        services.append("chat")
    return services


def _service_model_path(ds: DeploySettings, service: str) -> Path | None:
    return {
        "input": ds.input_model_path,
        "output": ds.output_model_path,
        "embed": ds.embed_model_path,
        "chat": ds.chat_model_path,
    }.get(service)


def _validate(ds: DeploySettings, services: list[str]) -> list[str]:
    """返回缺失模型路径的错误信息（空列表表示通过）；不抛异常，便于 --dry-run 只告警。"""
    errors: list[str] = []
    for service in services:
        path = _service_model_path(ds, service)
        if path is None:
            continue
        if not path.exists():
            errors.append(f"服务 {service} 的模型路径不存在：{path}")
    return errors


def derive_child_env(ds: DeploySettings, services: list[str]) -> dict[str, str]:
    """按端口派生各服务连接 URL，注入子进程环境（覆盖 .env / configs/services.env 中的对应值）。

    - 始终导出 ``APP_PORT`` 与输入/输出检测的 ``*_API_URL``。
    - ``EMBED_API_URL`` / ``CHAT_API_URL`` 仅当实际部署对应服务时覆盖，
      避免盖掉 .env 里配置的外部 Embedding / 云端聊天连接。
    - ``gpu_ids`` 经各子服务的 ``--gpu-ids`` 传递，不在此注入。
    """
    env = dict(os.environ)
    host = ds.url_host
    env["APP_PORT"] = str(ds.defense_port)
    env["INPUT_DETECTOR_API_URL"] = f"http://{host}:{ds.input_port}/v1"
    env["OUTPUT_DETECTOR_API_URL"] = f"http://{host}:{ds.output_port}/v1"
    if "embed" in services:
        env["EMBED_API_URL"] = f"http://{host}:{ds.embed_port}/v1"
    if "chat" in services:
        env["CHAT_API_URL"] = f"http://{host}:{ds.chat_port}/v1"
    return env


def build_commands(ds: DeploySettings, services: list[str], host: str) -> list[tuple[str, list[str]]]:
    """构造各服务的启动命令（复用现有 CLI 入口，``python -m ...``）。"""
    py = sys.executable
    commands: list[tuple[str, list[str]]] = []
    for service in services:
        if service == "defense":
            commands.append(("defense", [py, "-m", "injection_guard.cli.serve_defense"]))
        elif service in ("input", "output", "chat"):
            port = {"input": ds.input_port, "output": ds.output_port, "chat": ds.chat_port}[service]
            argv = [
                py, "-m", "injection_guard.cli.serve_llm",
                "--model_path", str(_service_model_path(ds, service)),
                "--host", host,
                "--port", str(port),
                "--workers", str(ds.workers),
                "--mode", ds.mode,
                "--request-timeout", str(ds.request_timeout),
                "--service", service,
            ]
            if ds.gpu_ids:
                argv += ["--gpu-ids", ds.gpu_ids]
            commands.append((service, argv))
        elif service == "embed":
            argv = [
                py, "-m", "injection_guard.cli.serve_embed",
                "--model_path", str(ds.embed_model_path),
                "--host", host,
                "--port", str(ds.embed_port),
                "--request-timeout", str(ds.request_timeout),
            ]
            if ds.gpu_ids:
                argv += ["--gpu-ids", ds.gpu_ids]
            commands.append(("embed", argv))
    return commands


def _print_dry_run(child_env: dict[str, str], commands: list[tuple[str, list[str]]]) -> None:
    print("== 派生环境变量（注入子进程，覆盖 .env / configs/services.env）==")
    for key in (
        "APP_PORT",
        "INPUT_DETECTOR_API_URL",
        "OUTPUT_DETECTOR_API_URL",
        "EMBED_API_URL",
        "CHAT_API_URL",
    ):
        if key in child_env:
            print(f"  {key}={child_env[key]}")
    print("== 启动命令 ==")
    for name, argv in commands:
        print(f"  [{name}] {' '.join(argv)}")


def _terminate_all(procs: list[tuple[str, subprocess.Popen]]) -> None:
    for _name, proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 10.0
    for _name, proc in procs:
        if proc.poll() is not None:
            continue
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()


def _run(commands: list[tuple[str, list[str]]], child_env: dict[str, str]) -> None:
    stop = threading.Event()

    def _handle_signal(signum: int, _frame: object) -> None:
        logger.info("收到信号 %s，停止全部服务", signum)
        stop.set()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        for name, argv in commands:
            logger.info("启动 %s：%s", name, " ".join(argv))
            procs.append((name, subprocess.Popen(argv, env=child_env)))
        while not stop.is_set():
            for name, proc in procs:
                rc = proc.poll()
                if rc is not None:
                    logger.warning("服务 %s 已退出（code=%s），停止其余服务", name, rc)
                    return
            stop.wait(0.5)
    finally:
        _terminate_all(procs)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="一键部署本地服务（读 configs/deploy.env）")
    parser.add_argument("--only", type=str, default=None, help="只部署指定服务（逗号分隔）：defense,input,output,embed,chat")
    parser.add_argument("--host", type=str, default=None, help="覆盖子服务绑定 Host（默认取 deploy.env 的 HOST）")
    parser.add_argument("--url-host", type=str, default=None, help="覆盖派生 URL 用的主机名（默认取 deploy.env 的 URL_HOST）")
    parser.add_argument("--dry-run", action="store_true", help="只打印派生环境变量与启动命令，不启动")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    setup_logging(get_settings().log_level)

    ds = get_deploy_settings()
    updates: dict[str, object] = {}
    if args.host:
        updates["host"] = args.host
    if args.url_host:
        updates["url_host"] = args.url_host
    if updates:
        ds = ds.model_copy(update=updates)

    services = _resolve_services(ds, args.only)
    errors = _validate(ds, services)
    if errors:
        for err in errors:
            logger.error("%s", err)
        if not args.dry_run:
            raise SystemExit(1)

    child_env = derive_child_env(ds, services)
    commands = build_commands(ds, services, ds.host)

    if args.dry_run:
        _print_dry_run(child_env, commands)
        return

    _run(commands, child_env)


if __name__ == "__main__":
    main()
