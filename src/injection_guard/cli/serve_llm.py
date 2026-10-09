"""启动 LLM 推理服务（OpenAI 兼容，支持 Logprobs）。"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import threading

from ..config import get_settings
from ..logging_config import setup_logging
from ..services.llm_server import start_api_server

logger = logging.getLogger(__name__)


def _start_cloudflared_quick_tunnel(local_port: int, local_host: str = "127.0.0.1") -> subprocess.Popen:
    cmd = [
        "cloudflared",
        "tunnel",
        "--url",
        f"http://{local_host}:{local_port}",
        "--no-autoupdate",
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    def _reader() -> None:
        public_url = None
        for line in proc.stdout:
            m = re.search(r"https://[a-zA-Z0-9\-]+\.trycloudflare\.com", line)
            if m and public_url is None:
                public_url = m.group(0)
                logger.info("[Cloudflare] Public URL: %s", public_url)

    threading.Thread(target=_reader, daemon=True).start()
    return proc


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM API Server Launcher (高并发)")

    # ── 模型参数 ──
    parser.add_argument("--mode", type=str, choices=["lora", "full"], default="full")
    parser.add_argument("--base_model_path", type=str, default=None, help="LoRA 模式下的基座模型路径")
    parser.add_argument(
        "--model_path", type=str, default="./finetune_wm", help="模型路径（full 或 lora 适配器路径）"
    )
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器路径（默认取 model_path）")

    # ── 网络参数 ──
    parser.add_argument("--host", type=str, default="127.1.1.1")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--public", action="store_true")

    # ── 高并发参数 ──
    parser.add_argument(
        "--workers", type=int, default=4, help="uvicorn worker 数量。多 GPU 场景手动设置，各 worker 独立加载模型"
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None, help="可用 GPU 列表，例如 0,1,2,3；默认使用全部可见 GPU"
    )
    parser.add_argument(
        "--request-timeout", type=int, default=300, help="单个推理请求的最大等待+执行时间（秒），默认 300"
    )

    args = parser.parse_args()

    setup_logging(get_settings().log_level)

    # GPU 列表需在首次 get_settings() 之前写入环境变量
    if args.gpu_ids:
        os.environ["API_SERVER_GPU_IDS"] = args.gpu_ids

    # 启动隧道（如果需要）
    tunnel_proc = None
    if args.public:
        try:
            tunnel_proc = _start_cloudflared_quick_tunnel(local_port=args.port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tunnel Error: %s", exc)

    try:
        start_api_server(
            host=args.host,
            port=args.port,
            workers=args.workers,
            request_timeout=args.request_timeout,
            mode=args.mode,
            model_path=args.model_path,
            base_model_path=args.base_model_path,
            lora_path=args.lora_path,
        )
    finally:
        if tunnel_proc:
            tunnel_proc.terminate()


if __name__ == "__main__":
    main()
