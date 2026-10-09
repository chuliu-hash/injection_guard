#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
API服务器模块 - 提供OpenAI API兼容的服务 (支持 Logprobs)
支持高并发：异步端点 + 线程池推理序列化 + 请求超时控制

高并发原理：
  - FastAPI 异步端点可以同时接受大量 HTTP 连接（不阻塞事件循环）
  - model.generate() 放入 ThreadPoolExecutor(max_workers=1) 执行，串行访问 GPU
  - 等待推理的请求自然排队，不会导致 connection refused
  - 每个请求有独立的超时控制，避免队列堆积

多 Worker 说明：
  workers=1（默认）: 父进程加载模型，单进程运行
  workers>1: 每个 uvicorn 子进程首次收到请求时独立加载模型（lazy loading），
            各自占用独立的显存。需确保每张 GPU 显存能容纳完整模型。
"""

import asyncio
import gc
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from typing import List, Optional
import time
import torch
import math
from concurrent.futures import ThreadPoolExecutor
from gpu_utils import assign_process_gpu, parse_gpu_ids

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


# --- Data Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class LogprobContent(BaseModel):
    token: str
    logprob: float

class ChoiceLogprobs(BaseModel):
    content: List[LogprobContent]

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "llama2-finetuned"
    max_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    logprobs: Optional[ChoiceLogprobs] = None
    finish_reason: str = "stop"

class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


# ── Module-level globals ──────────────────────────────────────
# 每个 uvicorn worker 进程独立拥有一份副本
_model = None
_tokenizer = None
_model_executor: Optional[ThreadPoolExecutor] = None
_request_timeout: float = 300.0
_model_config: dict = {}  # 模型加载配置（由 configure_model 设置）
_load_lock = None
_completed_requests = 0

app = FastAPI(title="LLM API with Logprobs", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 模型配置与懒加载 ──

def _get_int_env(name: str, default: int, min_value: Optional[int] = None) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if min_value is not None:
        value = max(min_value, value)
    return value


def _get_bool_env(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "y", "on")


def _get_float_env(name: str, default: Optional[float] = None) -> Optional[float]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _maybe_empty_cuda_cache(force: bool = False) -> None:
    global _completed_requests
    if not torch.cuda.is_available():
        return

    _completed_requests += 1
    every = _get_int_env("API_SERVER_EMPTY_CACHE_EVERY", 32, 0)
    should_clear = force or _get_bool_env("API_SERVER_EMPTY_CACHE_AFTER_REQUEST", False)
    should_clear = should_clear or (every > 0 and _completed_requests % every == 0)
    if should_clear:
        gc.collect()
        torch.cuda.empty_cache()

def configure_model(mode='full', model_path=None, base_model_path=None,
                    lora_path=None, request_timeout=300):
    """
    存储模型加载配置。各 worker 首次请求时按此配置懒加载模型。
    支持单 worker（父进程加载）和多 worker（各自加载）。

    多 worker 模式下，配置通过环境变量传递给子进程（避免 fork/spawn 丢失）。
    """
    # 环境变量方式：子进程无论 fork/spawn 都能继承
    os.environ['API_SERVER_MODE'] = mode or 'full'
    if model_path:
        os.environ['API_SERVER_MODEL_PATH'] = model_path
    if base_model_path:
        os.environ['API_SERVER_BASE_MODEL_PATH'] = base_model_path
    if lora_path:
        os.environ['API_SERVER_LORA_PATH'] = lora_path
    os.environ['API_SERVER_TIMEOUT'] = str(request_timeout or 300)

    global _model_config, _request_timeout
    _model_config = {
        'mode': mode,
        'model_path': model_path,
        'base_model_path': base_model_path,
        'lora_path': lora_path,
    }
    _request_timeout = request_timeout


def _ensure_model_loaded():
    """
    确保当前进程已加载模型。
    请求入口的懒加载路径：读取配置 → 加载模型。
    """
    global _model, _tokenizer, _model_executor, _model_config, _request_timeout, _load_lock

    if _model is not None:
        return

    if _load_lock is None:
        import threading
        _load_lock = threading.Lock()

    with _load_lock:
        if _model is not None:
            return

        # 环境变量回退：多 worker 子进程通过 os.environ 继承配置
        if not _model_config and os.environ.get('API_SERVER_MODEL_PATH'):
            _model_config = {
                'mode': os.environ.get('API_SERVER_MODE', 'full'),
                'model_path': os.environ.get('API_SERVER_MODEL_PATH'),
                'base_model_path': os.environ.get('API_SERVER_BASE_MODEL_PATH'),
                'lora_path': os.environ.get('API_SERVER_LORA_PATH'),
            }
            _request_timeout = float(os.environ.get('API_SERVER_TIMEOUT', '300'))

        if not _model_config or not _model_config.get('model_path'):
            raise RuntimeError("模型未配置，请先调用 configure_model()")

        _load_model()


def _select_worker_device():
    if not torch.cuda.is_available():
        return None

    device_count = torch.cuda.device_count()
    gpu_ids = parse_gpu_ids(os.environ.get("API_SERVER_GPU_IDS"), device_count)
    namespace = os.environ.get("API_SERVER_GPU_NAMESPACE", f"api_server_{os.getppid()}")
    gpu_id = assign_process_gpu(namespace, device_count, gpu_ids)
    if gpu_id is None:
        return None
    torch.cuda.set_device(gpu_id)
    memory_fraction = _get_float_env("API_SERVER_MEMORY_FRACTION")
    if memory_fraction is not None and 0 < memory_fraction <= 1:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)
    return gpu_id


def _load_model():
    """实际执行模型加载（同步，在 worker 启动时或首次请求时调用）。"""
    global _model, _tokenizer, _model_executor

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    pid = os.getpid()
    print(f"[Worker {pid}] 开始加载模型 ...")

    cfg = _model_config
    model_path = cfg.get('model_path')
    split_across_gpus = os.environ.get("API_SERVER_SPLIT_ACROSS_GPUS", "false").lower() == "true"
    gpu_id = None if split_across_gpus else _select_worker_device()
    device_map = "auto" if split_across_gpus else ({"": f"cuda:{gpu_id}"} if gpu_id is not None else None)
    if gpu_id is not None:
        print(f"[Worker {pid}] 绑定 GPU cuda:{gpu_id}")
    elif split_across_gpus:
        print(f"[Worker {pid}] 使用 device_map=auto 跨 GPU 切分模型")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg['mode'] == 'lora':
        base_path = cfg.get('base_model_path')
        if not base_path:
            raise RuntimeError("LoRA 模式需要 base_model_path")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        lora_path = cfg.get('lora_path', model_path)
        model = PeftModel.from_pretrained(model, lora_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )

    model.eval()
    _model = model
    _tokenizer = tokenizer
    _model_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="model_infer"
    )

    if not tokenizer.chat_template:
        print(f"[Worker {pid}] ⚠️ 当前 Tokenizer 没有默认的 chat_template")

    print(f"[Worker {pid}] 模型加载完成: {model_path}")


# ── 事件 ──

@app.on_event("shutdown")
async def _shutdown():
    global _model_executor
    if _model_executor:
        _model_executor.shutdown(wait=True)
        _model_executor = None


@app.on_event("startup")
def _load_on_startup():
    """
    Worker 进程启动时自动加载模型。
    多 worker 模式下每个子进程启动时都会执行一次，无需等待首个请求。
    """
    global _model_config, _request_timeout

    # 单 worker 模式已在 start_api_server 中预加载，跳过
    if _model is not None:
        return

    # 从环境变量读取配置（多 worker 子进程通过此方式继承）
    if not _model_config and os.environ.get('API_SERVER_MODEL_PATH'):
        _model_config = {
            'mode': os.environ.get('API_SERVER_MODE', 'full'),
            'model_path': os.environ.get('API_SERVER_MODEL_PATH'),
            'base_model_path': os.environ.get('API_SERVER_BASE_MODEL_PATH'),
            'lora_path': os.environ.get('API_SERVER_LORA_PATH'),
        }
        _request_timeout = float(os.environ.get('API_SERVER_TIMEOUT', '300'))

    if not _model_config or not _model_config.get('model_path'):
        print(f"[Worker {os.getpid()}] 模型未配置，跳过预加载")
        return

    try:
        _load_model()
        print(f"[Worker {os.getpid()}] 模型预加载完成")
    except Exception as e:
        print(f"[Worker {os.getpid()}] 模型预加载失败（首次请求会重试）: {e}")


# ── 健康检查 ──

@app.get("/health")
async def health():
    executor = _model_executor
    queue_size = 0
    if executor:
        queue_size = executor._work_queue.qsize() if hasattr(executor, '_work_queue') else -1
    cuda_memory = None
    if torch.cuda.is_available() and _model is not None:
        device = torch.cuda.current_device()
        cuda_memory = {
            "device": device,
            "allocated_mib": round(torch.cuda.memory_allocated(device) / 1024 / 1024, 2),
            "reserved_mib": round(torch.cuda.memory_reserved(device) / 1024 / 1024, 2),
            "max_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2),
        }
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "request_timeout": _request_timeout,
        "infer_queue_size": queue_size,
        "worker_pid": os.getpid(),
        "cuda_device": torch.cuda.current_device() if torch.cuda.is_available() and _model is not None else None,
        "cuda_memory": cuda_memory,
    }


# ── 核心推理端点 ──

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """
    异步推理端点。
    首次请求时会触发当前 worker 的模型加载（lazy loading）。
    """
    # 确保当前 worker 已加载模型
    try:
        _ensure_model_loaded()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        max_queue_size = _get_int_env("API_SERVER_MAX_QUEUE_SIZE", 32, 0)
        if max_queue_size > 0 and _model_executor is not None:
            queue_size = _model_executor._work_queue.qsize() if hasattr(_model_executor, '_work_queue') else 0
            if queue_size >= max_queue_size:
                raise HTTPException(
                    status_code=503,
                    detail=f"推理队列已满（{queue_size}/{max_queue_size}），请稍后重试"
                )

        loop = asyncio.get_event_loop()

        result = await asyncio.wait_for(
            loop.run_in_executor(
                _model_executor,
                _chat_completion_sync,
                request
            ),
            timeout=_request_timeout
        )
        return result

    except asyncio.TimeoutError:
        _maybe_empty_cuda_cache(force=True)
        raise HTTPException(
            status_code=504,
            detail=f"推理队列超时（{_request_timeout}s），请稍后重试"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 同步推理函数（在线程池中执行） ──

def _chat_completion_sync(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """同步执行模型推理（在 ThreadPoolExecutor 线程池中运行）。"""
    model = _model
    tokenizer = _tokenizer
    max_input_tokens = _get_int_env("API_SERVER_MAX_INPUT_TOKENS", 2048, 1)
    max_new_tokens_limit = _get_int_env("API_SERVER_MAX_NEW_TOKENS", 256, 1)

    # 1. Chat Template
    try:
        prompt = tokenizer.apply_chat_template(
            request.messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception as e:
        print(f"Warning: apply_chat_template failed ({e}), falling back to simple concatenation.")
        prompt = request.messages[-1].content

    # 2. 编码
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_tokens,
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    input_length = inputs.input_ids.shape[1]

    # 3. 生成配置
    requested_max_tokens = request.max_tokens or max_new_tokens_limit
    max_new_tokens = min(requested_max_tokens, max_new_tokens_limit)
    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9
    do_sample = temperature > 0.0
    from transformers import GenerationConfig

    generation_config = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=request.logprobs,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    generation_kwargs = {
        **inputs,
        "generation_config": generation_config,
    }

    # 4. 生成
    try:
        with torch.inference_mode():
            outputs = model.generate(**generation_kwargs)
    except Exception as e:
        print(f"Generation Error: {e}")
        _maybe_empty_cuda_cache(force=True)
        raise

    # 5. 解码
    generated_ids = outputs.sequences[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 6. Logprobs
    logprobs_data = None
    if request.logprobs and outputs.scores:
        try:
            transition_scores = model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            content_list = []
            for i, score in enumerate(transition_scores[0]):
                token_id = generated_ids[i]
                token_text = tokenizer.decode([token_id])
                content_list.append(LogprobContent(
                    token=token_text,
                    logprob=float(score.cpu().item())
                ))
            logprobs_data = ChoiceLogprobs(content=content_list)
        except Exception:
            pass

    # 7. 响应
    try:
        return ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time() * 1000)}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    logprobs=logprobs_data,
                    finish_reason="stop"
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_length,
                completion_tokens=len(generated_ids),
                total_tokens=input_length + len(generated_ids)
            )
        )
    finally:
        del inputs, outputs, generated_ids
        _maybe_empty_cuda_cache()


# ── 启动入口 ──

def start_api_server(host="0.0.0.0", port=8000, workers=1, request_timeout=300,
                     mode='full', model_path=None, base_model_path=None, lora_path=None):
    """
    启动高并发 OpenAI API 兼容服务。

    参数：
        host, port: 监听地址
        workers: uvicorn worker 数量。多 GPU 场景可>1，各 worker 独占显存
        request_timeout: 单个请求的最大等待+推理时间（秒）
        mode: 'full' 或 'lora'
        model_path: 模型路径（full mode 或 lora 的适配器路径）
        base_model_path: LoRA 模式下的基座模型路径
        lora_path: LoRA 适配器路径（默认取 model_path）
    """
    configure_model(
        mode=mode,
        model_path=model_path,
        base_model_path=base_model_path,
        lora_path=lora_path,
        request_timeout=request_timeout,
    )

    # workers=1 时预加载模型（首次请求不等待）
    if workers == 1:
        try:
            _ensure_model_loaded()
        except Exception as e:
            print(f"⚠️  模型预加载失败（首次请求时会重试）: {e}")

    if workers > 1:
        print(f"\n⚠️  多 worker 模式 (workers={workers})")
        print(f"   每个 uvicorn 子进程会在首次请求时独立加载模型。")
        print(f"   请确保各 GPU 显存能容纳完整模型。\n")

    print(f"🚀 高并发推理服务启动: http://{host}:{port}")
    print(f"   workers={workers}, request_timeout={request_timeout}s")

    if workers > 1:
        uvicorn.run("api_server:app", host=host, port=port, workers=workers)
    else:
        uvicorn.run(app, host=host, port=port)
