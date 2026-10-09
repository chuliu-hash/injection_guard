"""OpenAI 兼容的 Embedding API 服务（高并发版）。

支持高并发：异步端点不阻塞事件循环，``embedder.encode()`` 在线程池中执行，
并支持 uvicorn 多 worker。
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from ..config import get_settings
from .gpu import assign_process_gpu, parse_gpu_ids

logger = logging.getLogger(__name__)


# ── 进程级状态 ────────────────────────────────────────────────────────
_embedder: Optional[SentenceTransformer] = None
_embed_device: Optional[str] = None
_embedder_lock = threading.Lock()
_model_path: Optional[str] = None

# 线程池：embedding 计算放入线程池以释放事件循环
_embed_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed_infer")


# ── 数据结构（完全兼容 OpenAI）────────────────────────────────────────
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = Field(default="embedding-model")


class EmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingData]
    model: str
    usage: Usage


def _select_worker_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"

    settings = get_settings()
    device_count = torch.cuda.device_count()
    gpu_ids = parse_gpu_ids(settings.embed_gpu_ids, device_count)
    namespace = settings.embed_gpu_namespace or f"embed_{os.getppid()}"
    gpu_id = assign_process_gpu(namespace, device_count, gpu_ids)
    if gpu_id is None:
        return "cpu"
    torch.cuda.set_device(gpu_id)
    return f"cuda:{gpu_id}"


def _ensure_embedder_loaded() -> SentenceTransformer:
    global _embedder, _embed_device
    if _embedder is not None:
        return _embedder

    with _embedder_lock:
        if _embedder is not None:
            return _embedder

        _embed_device = _select_worker_device()
        logger.info("正在加载 Embedding 模型 [%s] 到 %s ...", _model_path, _embed_device)
        _embedder = SentenceTransformer(
            _model_path,
            trust_remote_code=True,
            device=_embed_device,
            model_kwargs={"torch_dtype": torch.bfloat16},
        )
        logger.info("Embedding 模型加载完成: device=%s", _embed_device)
        return _embedder


def _sync_embed(request: EmbeddingRequest) -> EmbeddingResponse:
    """同步执行 embedding 编码（在线程池中运行）。"""
    local_embedder = _ensure_embedder_loaded()
    texts = [request.input] if isinstance(request.input, str) else request.input

    embeddings = local_embedder.encode(texts, convert_to_tensor=True)
    embeddings_list = embeddings.cpu().float().tolist()

    response_data = [
        EmbeddingData(index=i, embedding=emb) for i, emb in enumerate(embeddings_list)
    ]

    approx_tokens = sum(len(t) for t in texts) // 4

    return EmbeddingResponse(
        data=response_data,
        model=request.model,
        usage=Usage(prompt_tokens=approx_tokens, total_tokens=approx_tokens),
    )


def create_app(model_path: str, request_timeout: float = 60.0) -> FastAPI:
    """构造 Embedding 服务 app（``model_path`` 在首个请求时懒加载）。"""
    global _model_path
    _model_path = model_path

    app = FastAPI(title="OpenAI-Compatible Embedding API (高并发)")

    @app.on_event("startup")
    def _startup() -> None:
        _ensure_embedder_loaded()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        _embed_executor.shutdown(wait=True)

    # --- OpenAI 标准路由（异步）---
    @app.post("/v1/embeddings", response_model=EmbeddingResponse)
    async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
        """异步 Embedding 端点，不阻塞事件循环。"""
        try:
            loop = asyncio.get_event_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(_embed_executor, _sync_embed, request),
                timeout=request_timeout,
            )
            return result
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Embedding 推理超时") from None
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # --- 健康检查 ---
    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model": _model_path,
            "worker_pid": os.getpid(),
            "device": _embed_device,
        }

    return app


def run_server(host: str, port: int, model_path: str, workers: int = 1, request_timeout: float = 60.0) -> None:
    """启动 Embedding 服务。"""
    logger.info("OpenAI 兼容的 Embedding API 启动: %s:%s (workers=%s)", host, port, workers)
    if workers > 1:
        logger.warning("多 worker 时每个进程会加载独立的模型副本")
        os.environ["EMBED_MODEL_PATH"] = model_path
        uvicorn.run(
            "injection_guard.services.embed_server:_factory",
            factory=True,
            host=host,
            port=port,
            workers=workers,
        )
    else:
        uvicorn.run(create_app(model_path, request_timeout), host=host, port=port)


def _factory() -> FastAPI:
    """供多 worker 模式（``uvicorn --factory``）使用的零参工厂，读取环境变量。"""
    settings = get_settings()
    return create_app(
        model_path=os.environ.get("EMBED_MODEL_PATH", settings.embed_model_name),
        request_timeout=settings.embed_request_timeout,
    )
