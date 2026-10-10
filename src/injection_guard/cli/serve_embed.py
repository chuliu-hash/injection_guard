"""启动 Embedding 服务（OpenAI 兼容）。"""

from __future__ import annotations

import argparse
import os

from ..config import get_settings
from ..logging_config import setup_logging
from ..services.embed_server import run_server


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-Compatible Embedding API Server (高并发)")
    parser.add_argument(
        "--model_path", type=str, default="Qwen3-Embedding-0.6B", help="HuggingFace 模型名或本地路径"
    )
    parser.add_argument("--port", type=int, default=8000, help="API 服务端口（默认 8000）")
    parser.add_argument("--host", type=str, default="localhost", help="绑定的 Host")
    parser.add_argument(
        "--workers", type=int, default=1, help="uvicorn worker 数量。单 GPU 保持 1（默认），多 GPU 可增加"
    )
    parser.add_argument(
        "--gpu-ids", type=str, default=None, help="可用 GPU 列表，例如 0,1,2,3；默认使用全部可见 GPU"
    )
    parser.add_argument("--request-timeout", type=int, default=60, help="单个推理请求超时（秒），默认 60")

    args = parser.parse_args()

    # GPU 列表需在首次 get_settings() 之前写入环境变量（@lru_cache 冻结首次读取）；
    # 勿在此之前读取应用配置（例如为日志提前调用 get_settings）。
    if args.gpu_ids:
        os.environ["EMBED_GPU_IDS"] = args.gpu_ids

    setup_logging(get_settings().log_level)

    run_server(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        workers=args.workers,
        request_timeout=args.request_timeout,
    )


if __name__ == "__main__":
    main()
