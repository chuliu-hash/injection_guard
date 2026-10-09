"""
OpenAI 兼容的 Embedding API 服务器（高并发版）

支持高并发：
  - 异步端点不阻塞事件循环
  - embedder.encode() 在线程池中执行
  - 支持 uvicorn 多 worker

用法：
    python interact_embed.py --port 8090 --workers 1
"""
import argparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Union, List
from sentence_transformers import SentenceTransformer
import torch
import uvicorn
import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from gpu_utils import assign_process_gpu, parse_gpu_ids


# ==========================================
# 1. 解析命令行参数
# ==========================================
parser = argparse.ArgumentParser(description="OpenAI-Compatible Embedding API Server (高并发)")
parser.add_argument(
    "--model_path",
    type=str,
    default="Qwen3-Embedding-0.6B",
    help="指定 HuggingFace 模型名称或本地绝对路径"
)
parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="指定 API 服务运行的端口 (默认: 8000)"
)
parser.add_argument(
    "--host",
    type=str,
    default="127.1.1.1",
    help="指定绑定的 Host"
)
parser.add_argument(
    "--workers", type=int, default=1,
    help="uvicorn worker 数量。单 GPU 保持 1（默认），多 GPU 可适当增加"
)
parser.add_argument(
    "--gpu-ids", type=str, default=None,
    help="可用 GPU 列表，例如 0,1,2,3；默认使用全部可见 GPU"
)
parser.add_argument(
    "--request-timeout", type=int, default=60,
    help="单个推理请求超时（秒），默认 60"
)
args = parser.parse_args()
if args.gpu_ids:
    os.environ["EMBED_GPU_IDS"] = args.gpu_ids

# ==========================================
# 2. 初始化 FastAPI 与 加载模型
# ==========================================
app = FastAPI(title="OpenAI-Compatible Embedding API (高并发)")

embedder = None
_embedder_lock = threading.Lock()
_embed_device = None

# 线程池：embedding 计算放入线程池以释放事件循环
_embed_executor = ThreadPoolExecutor(
    max_workers=1,
    thread_name_prefix="embed_infer"
)

# --- 数据结构定义 (完全兼容 OpenAI) ---
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = Field(default=args.model_path)

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


def _select_worker_device():
    if not torch.cuda.is_available():
        return "cpu"

    device_count = torch.cuda.device_count()
    gpu_ids = parse_gpu_ids(os.environ.get("EMBED_GPU_IDS"), device_count)
    namespace = os.environ.get("EMBED_GPU_NAMESPACE", f"embed_{os.getppid()}")
    gpu_id = assign_process_gpu(namespace, device_count, gpu_ids)
    if gpu_id is None:
        return "cpu"
    torch.cuda.set_device(gpu_id)
    return f"cuda:{gpu_id}"


def _ensure_embedder_loaded():
    global embedder, _embed_device
    if embedder is not None:
        return embedder

    with _embedder_lock:
        if embedder is not None:
            return embedder

        _embed_device = _select_worker_device()
        print(f"🚀 正在加载 Embedding 模型 [{args.model_path}] 到 {_embed_device}...")
        embedder = SentenceTransformer(
            args.model_path,
            trust_remote_code=True,
            device=_embed_device,
            model_kwargs={"torch_dtype": torch.bfloat16}
        )
        print(f"✅ Embedding 模型加载完成: device={_embed_device}")
        return embedder


@app.on_event("startup")
def startup():
    _ensure_embedder_loaded()


def _sync_embed(request: EmbeddingRequest) -> EmbeddingResponse:
    """同步执行 embedding 编码（在线程池中运行）"""
    local_embedder = _ensure_embedder_loaded()
    texts = [request.input] if isinstance(request.input, str) else request.input

    embeddings = local_embedder.encode(texts, convert_to_tensor=True)
    embeddings_list = embeddings.cpu().float().tolist()

    response_data = []
    for i, emb in enumerate(embeddings_list):
        response_data.append(EmbeddingData(index=i, embedding=emb))

    approx_tokens = sum(len(t) for t in texts) // 4

    return EmbeddingResponse(
        data=response_data,
        model=request.model,
        usage=Usage(prompt_tokens=approx_tokens, total_tokens=approx_tokens)
    )


# --- OpenAI 标准路由（异步） ---
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embeddings(request: EmbeddingRequest):
    """异步 Embedding 端点，不阻塞事件循环"""
    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(_embed_executor, _sync_embed, request),
            timeout=args.request_timeout
        )
        return result
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Embedding 推理超时")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 健康检查 ---
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": args.model_path,
        "worker_pid": os.getpid(),
        "device": _embed_device,
    }


# --- 优雅关闭 ---
@app.on_event("shutdown")
async def shutdown():
    _embed_executor.shutdown(wait=True)


# ==========================================
# 3. 启动服务
# ==========================================
if __name__ == "__main__":
    print(f"✅ OpenAI 兼容的 Embedding API 准备就绪！")
    print(f"   监听: {args.host}:{args.port}")
    print(f"   workers: {args.workers}")
    print(f"   GPU列表: {args.gpu_ids or '全部可见GPU'}")
    if args.workers > 1:
        print(f"   ⚠️  多 worker 时每个进程会加载独立的模型副本")
    if args.workers > 1:
        uvicorn.run("interact_embed:app", host=args.host, port=args.port, workers=args.workers)
    else:
        uvicorn.run(app, host=args.host, port=args.port)
