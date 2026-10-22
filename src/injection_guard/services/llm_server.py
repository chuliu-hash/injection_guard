"""OpenAI API 兼容的本地 LLM 推理服务（支持 Logprobs 与 response_format）。

高并发原理：
  - FastAPI 异步端点可同时接受大量 HTTP 连接（不阻塞事件循环）；
  - ``model.generate()`` 放入 ``ThreadPoolExecutor(max_workers=1)`` 串行访问 GPU；
  - 等待推理的请求自然排队，不会 connection refused；
  - 每个请求有独立超时，避免队列堆积。

多 Worker：``workers>1`` 时每个 uvicorn 子进程首次收到请求时独立加载模型
（懒加载），各自占用独立显存；需确保每张 GPU 显存能容纳完整模型。
"""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..config import InferenceSettings, get_inference_settings
from .gpu import assign_process_gpu, parse_gpu_ids, resolve_namespace

logger = logging.getLogger(__name__)

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
    # OpenAI 现行字段名（langchain-openai 等新客户端发的是这个）；与 max_tokens 等价，
    # 两者同时出现时以 max_completion_tokens 为准。
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    # OpenAI 兼容的采样参数。本地实现把正向 penalty 近似映射为 repetition_penalty；
    # seed 用于固定采样随机源，便于复现。远程服务按各厂商语义执行。
    frequency_penalty: Optional[float] = 0.0
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = None
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    # OpenAI 兼容的「结构化输出」参数（langchain 的 with_structured_output 会发它）：
    #   {"type": "json_object"} 或
    #   {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}, "strict": true}}
    # 本地推理没有 logits 级约束解码（不想为一个功能引入 xgrammar / lm-format-enforcer
    # 这类新依赖），这里等价实现为「注入 JSON 指令 -> 生成 -> 校验 -> 不合法就重试一次」，
    # 见 _json_instruction / _json_text。**不保证** 100% 合法，客户端仍需解析兜底。
    response_format: Optional[Dict[str, Any]] = None

    def effective_max_tokens(self) -> Optional[int]:
        """请求体的生成长度上限（兼容 max_tokens / max_completion_tokens 两种字段名）。"""
        if self.max_completion_tokens is not None:
            return self.max_completion_tokens
        return self.max_tokens


def _json_instruction(response_format: Optional[Dict[str, Any]], *, strict: bool = False) -> Optional[str]:
    """把 ``response_format`` 翻译成给本地模型的「只输出 JSON」指令；不需要则返回 None。

    ``strict=True`` 用于生成结果不是合法 JSON 后的重试，措辞更强硬。
    """
    if not response_format:
        return None
    fmt = str(response_format.get("type") or "").lower()
    if fmt not in {"json_object", "json_schema"}:
        return None

    parts = ["你必须只输出一个合法的 JSON 值，不要输出任何解释、Markdown 代码块标记或多余文字。"]
    if fmt == "json_schema":
        spec = response_format.get("json_schema") or {}
        schema = spec.get("schema") if isinstance(spec, dict) else None
        if schema is None and isinstance(spec, dict):
            schema = spec  # 少数客户端直接把 schema 放在 json_schema 下
        if schema:
            parts.append("输出的 JSON 必须严格符合以下 JSON Schema：")
            parts.append(json.dumps(schema, ensure_ascii=False))
    if strict:
        parts.append("上一次输出不是合法 JSON。这次必须从第一个字符就是 { 或 [，并输出完整闭合的 JSON。")
    return "\n".join(parts)


def _json_text(text: str) -> Optional[str]:
    """从模型回复里取出 JSON 文本；取不到返回 None。

    容忍常见的 ```json 代码块包裹与前后寒暄（取第一个 ``{``/``[`` 到最后一个 ``}``/``]``）。
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    if not stripped:
        return None

    starts = [index for index in (stripped.find("{"), stripped.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    end = max(stripped.rfind("}"), stripped.rfind("]"))
    if end <= start:
        return None
    return stripped[start : end + 1]


def _is_valid_json(text: str) -> bool:
    """判断模型输出能否解析成 JSON（用于决定要不要重试一次）。"""
    candidate = _json_text(text)
    if candidate is None:
        return False
    try:
        json.loads(candidate)
    except (TypeError, ValueError):
        return False
    return True


def _messages_with_json_instruction(request: ChatCompletionRequest) -> List[ChatMessage]:
    """在对话末尾追加「只输出 JSON」指令（``response_format`` 缺省时原样返回）。

    指令作为独立的 user 消息追加在末尾，而不是拼进 chat template 的前缀，
    以免破坏 Qwen 等模板对 ``<|im_start|>`` 角色的依赖。
    """
    instruction = _json_instruction(request.response_format)
    if instruction is None:
        return list(request.messages)
    return [*request.messages, ChatMessage(role="user", content=instruction)]


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
_service: Optional[str] = None  # 服务名（input/output/chat），决定读取 configs/services.env 的前缀


def _inference_settings() -> InferenceSettings:
    """取当前服务的推理配置（按服务前缀）；多 worker 子进程经环境变量继承服务名。"""
    return get_inference_settings(_service or os.environ.get("API_SERVER_SERVICE"))


app = FastAPI(title="LLM API with Logprobs", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 模型配置与懒加载 ──


def _maybe_empty_cuda_cache(force: bool = False) -> None:
    global _completed_requests
    if not torch.cuda.is_available():
        return

    settings = _inference_settings()
    _completed_requests += 1
    every = max(0, settings.api_server_empty_cache_every)
    should_clear = force or settings.api_server_empty_cache_after_request
    should_clear = should_clear or (every > 0 and _completed_requests % every == 0)
    if should_clear:
        gc.collect()
        torch.cuda.empty_cache()


def configure_model(
    mode: str = "full",
    model_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    request_timeout: float = 300,
    service: Optional[str] = None,
) -> None:
    """存储模型加载配置。各 worker 首次请求时按此配置懒加载模型。

    多 worker 模式下，配置通过环境变量传递给子进程（避免 fork/spawn 丢失）。
    ``service`` 决定推理参数从 ``configs/services.env`` 读取的前缀（input/output/chat）。
    """
    # 环境变量方式：子进程无论 fork/spawn 都能继承
    os.environ["API_SERVER_MODE"] = mode or "full"
    if model_path:
        os.environ["API_SERVER_MODEL_PATH"] = model_path
    if base_model_path:
        os.environ["API_SERVER_BASE_MODEL_PATH"] = base_model_path
    if lora_path:
        os.environ["API_SERVER_LORA_PATH"] = lora_path
    os.environ["API_SERVER_TIMEOUT"] = str(request_timeout or 300)
    if service:
        os.environ["API_SERVER_SERVICE"] = service

    global _model_config, _request_timeout, _service
    _model_config = {
        "mode": mode,
        "model_path": model_path,
        "base_model_path": base_model_path,
        "lora_path": lora_path,
    }
    _request_timeout = request_timeout
    _service = service


def _ensure_model_loaded() -> None:
    """确保当前进程已加载模型（请求入口的懒加载路径）。"""
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
        if not _model_config and os.environ.get("API_SERVER_MODEL_PATH"):
            _model_config = {
                "mode": os.environ.get("API_SERVER_MODE", "full"),
                "model_path": os.environ.get("API_SERVER_MODEL_PATH"),
                "base_model_path": os.environ.get("API_SERVER_BASE_MODEL_PATH"),
                "lora_path": os.environ.get("API_SERVER_LORA_PATH"),
            }
            _request_timeout = float(os.environ.get("API_SERVER_TIMEOUT", "300"))

        if not _model_config or not _model_config.get("model_path"):
            raise RuntimeError("模型未配置，请先调用 configure_model()")

        _load_model()


def _select_worker_device() -> Optional[int]:
    if not torch.cuda.is_available():
        return None

    settings = _inference_settings()
    device_count = torch.cuda.device_count()
    gpu_ids = parse_gpu_ids(settings.api_server_gpu_ids, device_count)
    namespace = resolve_namespace(settings.api_server_gpu_namespace)
    gpu_id = assign_process_gpu(namespace, device_count, gpu_ids)
    if gpu_id is None:
        return None
    torch.cuda.set_device(gpu_id)
    memory_fraction = settings.api_server_memory_fraction
    if memory_fraction is not None and 0 < memory_fraction <= 1:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, gpu_id)
    return gpu_id


def _load_model() -> None:
    """实际执行模型加载（同步，在 worker 启动时或首次请求时调用）。"""
    global _model, _tokenizer, _model_executor

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    settings = _inference_settings()
    pid = os.getpid()
    logger.info("[Worker %s] 开始加载模型 ...", pid)

    cfg = _model_config
    model_path = cfg.get("model_path")
    split_across_gpus = settings.api_server_split_across_gpus
    gpu_id = None if split_across_gpus else _select_worker_device()
    device_map = "auto" if split_across_gpus else ({"": f"cuda:{gpu_id}"} if gpu_id is not None else None)
    if gpu_id is not None:
        logger.info("[Worker %s] 绑定 GPU cuda:%s", pid, gpu_id)
    elif split_across_gpus:
        logger.info("[Worker %s] 使用 device_map=auto 跨 GPU 切分模型", pid)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["mode"] == "lora":
        base_path = cfg.get("base_model_path")
        if not base_path:
            raise RuntimeError("LoRA 模式需要 base_model_path")
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True,
        )
        lora_path = cfg.get("lora_path", model_path)
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
    _model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_infer")

    if not tokenizer.chat_template:
        logger.warning("[Worker %s] 当前 Tokenizer 没有默认的 chat_template", pid)

    logger.info("[Worker %s] 模型加载完成: %s", pid, model_path)


# ── 事件 ──


@app.on_event("shutdown")
async def _shutdown() -> None:
    global _model_executor
    if _model_executor:
        _model_executor.shutdown(wait=True)
        _model_executor = None


@app.on_event("startup")
def _load_on_startup() -> None:
    """Worker 进程启动时自动加载模型（多 worker 模式下每个子进程都会执行）。"""
    global _model_config, _request_timeout

    # 单 worker 模式已在 start_api_server 中预加载，跳过
    if _model is not None:
        return

    # 从环境变量读取配置（多 worker 子进程通过此方式继承）
    if not _model_config and os.environ.get("API_SERVER_MODEL_PATH"):
        _model_config = {
            "mode": os.environ.get("API_SERVER_MODE", "full"),
            "model_path": os.environ.get("API_SERVER_MODEL_PATH"),
            "base_model_path": os.environ.get("API_SERVER_BASE_MODEL_PATH"),
            "lora_path": os.environ.get("API_SERVER_LORA_PATH"),
        }
        _request_timeout = float(os.environ.get("API_SERVER_TIMEOUT", "300"))

    if not _model_config or not _model_config.get("model_path"):
        logger.info("[Worker %s] 模型未配置，跳过预加载", os.getpid())
        return

    try:
        _load_model()
        logger.info("[Worker %s] 模型预加载完成", os.getpid())
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Worker %s] 模型预加载失败（首次请求会重试）: %s", os.getpid(), exc)


# ── 健康检查 ──


@app.get("/health")
async def health() -> dict:
    executor = _model_executor
    queue_size = 0
    if executor:
        queue_size = executor._work_queue.qsize() if hasattr(executor, "_work_queue") else -1
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
        "cuda_device": torch.cuda.current_device()
        if torch.cuda.is_available() and _model is not None
        else None,
        "cuda_memory": cuda_memory,
    }


# ── 核心推理端点 ──


# SSE 响应头：禁用中间层缓冲，保证逐 token 实时下发
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """异步推理端点。首次请求时触发当前 worker 的模型加载（lazy loading）。

    ``stream=true`` 时返回 OpenAI 兼容的 SSE（``chat.completion.chunk``）；
    否则返回单次 JSON（``chat.completion``）。
    """
    # 确保当前 worker 已加载模型
    try:
        _ensure_model_loaded()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    try:
        max_queue_size = _inference_settings().api_server_max_queue_size
        if max_queue_size > 0 and _model_executor is not None:
            queue_size = (
                _model_executor._work_queue.qsize() if hasattr(_model_executor, "_work_queue") else 0
            )
            if queue_size >= max_queue_size:
                raise HTTPException(
                    status_code=503,
                    detail=f"推理队列已满（{queue_size}/{max_queue_size}），请稍后重试",
                )

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(request),
                media_type="text/event-stream",
                headers=_SSE_HEADERS,
            )

        loop = asyncio.get_event_loop()

        result = await asyncio.wait_for(
            loop.run_in_executor(_model_executor, _chat_completion_sync, request),
            timeout=_request_timeout,
        )
        return result

    except asyncio.TimeoutError:
        _maybe_empty_cuda_cache(force=True)
        raise HTTPException(
            status_code=504, detail=f"推理队列超时（{_request_timeout}s），请稍后重试"
        ) from None
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── 生成准备（同步 / 流式共用） ──


def _prepare_generation(
    request: ChatCompletionRequest,
    output_scores: Optional[bool] = None,
    messages: Optional[List[ChatMessage]] = None,
) -> tuple:
    """准备 chat template、输入编码与 ``GenerationConfig``。

    ``output_scores`` 为 ``None`` 时取 ``request.logprobs``（流式路径显式传 False）。
    ``messages`` 可覆盖请求自带的对话：默认在 ``response_format`` 存在时于末尾补一条
    JSON 指令（见 :func:`_messages_with_json_instruction`），重试时传入更严格的版本。
    返回 ``(prompt, inputs, input_length, generation_config)``。
    """
    model = _model
    tokenizer = _tokenizer
    settings = _inference_settings()
    max_input_tokens = settings.api_server_max_input_tokens
    # None = 不设上限（生成长度完全由请求的 max_tokens 决定）
    max_new_tokens_limit = settings.api_server_max_new_tokens

    # 1. Chat Template
    if messages is None:
        messages = _messages_with_json_instruction(request)
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("apply_chat_template 失败 (%s)，回退到简单拼接。", exc)
        prompt = messages[-1].content

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
    # 请求 max_tokens 优先；服务端上限（若设置）只作硬顶，None 表示不设上限
    requested_max_tokens = request.effective_max_tokens() or max_new_tokens_limit or 1024
    max_new_tokens = (
        min(requested_max_tokens, max_new_tokens_limit)
        if max_new_tokens_limit is not None
        else requested_max_tokens
    )
    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9
    do_sample = temperature > 0.0
    penalty = max(0.0, request.frequency_penalty or 0.0, request.presence_penalty or 0.0)
    if request.seed is not None:
        torch.manual_seed(request.seed)
    from transformers import GenerationConfig

    if output_scores is None:
        output_scores = bool(request.logprobs)

    generation_config = GenerationConfig(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=output_scores,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=min(2.0, 1.1 + penalty),
    )
    if do_sample:
        generation_config.temperature = temperature
        generation_config.top_p = top_p
    else:
        generation_config.temperature = None
        generation_config.top_p = None
        generation_config.top_k = None

    return prompt, inputs, input_length, generation_config


# ── 流式推理 ──


def _chunk_frame(
    completion_id: str,
    model_name: str,
    created: int,
    delta: dict,
    finish_reason: Optional[str] = None,
) -> str:
    """构造一帧 OpenAI ``chat.completion.chunk`` SSE 数据。"""
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _chat_completion_stream(request: ChatCompletionRequest) -> Iterator[str]:
    """逐 token 生成文本（阻塞生成器，由调用线程迭代）。

    ``model.generate`` 提交到 ``_model_executor``（max_workers=1）串行执行，
    以保证与非流式请求共享同一 GPU 串行语义；本函数在另一线程消费 ``streamer``。
    """
    from transformers import TextIteratorStreamer

    model = _model
    tokenizer = _tokenizer
    _prompt, inputs, _input_length, generation_config = _prepare_generation(
        request, output_scores=False
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = {**inputs, "generation_config": generation_config, "streamer": streamer}

    def _generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**generation_kwargs)
        finally:
            # 无论如何都结束流，避免消费线程永久阻塞
            streamer.end()

    future = _model_executor.submit(_generate)
    try:
        for token in streamer:
            if token:
                yield token
    finally:
        del inputs
        _maybe_empty_cuda_cache()

    error = future.exception()
    if error is not None:
        logger.error("流式生成异常: %s", error)
        raise error


async def _stream_chat_completion(request: ChatCompletionRequest):
    """把阻塞的流式生成桥接为 SSE（生成在 worker 线程，事件经 asyncio.Queue 回传）。"""
    completion_id = f"chatcmpl-{int(time.time() * 1000)}"
    created = int(time.time())
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()
    sentinel = object()
    error_box: dict = {}

    def _push(item: object) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            # 事件循环已关闭（客户端断开），静默丢弃
            pass

    def _worker() -> None:
        try:
            for token in _chat_completion_stream(request):
                _push(token)
        except Exception as exc:  # noqa: BLE001
            error_box["error"] = exc
        finally:
            _push(sentinel)

    thread = threading.Thread(target=_worker, name="llm_stream", daemon=True)
    thread.start()

    # 首帧：声明角色
    yield _chunk_frame(completion_id, request.model, created, {"role": "assistant", "content": ""})

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=_request_timeout)
        except asyncio.TimeoutError:
            logger.warning("流式推理超时（%ss 无新 token）", _request_timeout)
            break
        if item is sentinel:
            break
        yield _chunk_frame(completion_id, request.model, created, {"content": item})

    if error_box.get("error") is not None:
        logger.error("流式推理失败: %s", error_box["error"])

    yield _chunk_frame(completion_id, request.model, created, {}, finish_reason="stop")
    yield "data: [DONE]\n\n"


# ── 同步推理函数（在线程池中执行） ──


def _run_generation(
    request: ChatCompletionRequest, messages: Optional[List[ChatMessage]] = None
) -> Tuple[Any, str, Any, int, Any]:
    """执行一次同步生成。

    返回 ``(outputs, response_text, generated_ids, input_length, inputs)``；
    调用方负责在 ``finally`` 中释放 ``inputs`` / ``outputs`` 并清理显存缓存。
    """
    model = _model
    tokenizer = _tokenizer
    _prompt, inputs, input_length, generation_config = _prepare_generation(
        request, messages=messages
    )
    try:
        with torch.inference_mode():
            outputs = model.generate(**{**inputs, "generation_config": generation_config})
    except Exception:
        logger.exception("Generation Error")
        del inputs
        _maybe_empty_cuda_cache(force=True)
        raise

    generated_ids = outputs.sequences[0][input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return outputs, response_text, generated_ids, input_length, inputs


def _chat_completion_sync(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """同步执行模型推理（在 ThreadPoolExecutor 线程池中运行）。"""
    model = _model
    tokenizer = _tokenizer

    # 4. 生成（response_format 存在时等价实现为「JSON 指令 + 校验 + 重试一次」）
    outputs, response_text, generated_ids, input_length, inputs = _run_generation(request)
    if request.response_format and not _is_valid_json(response_text):
        logger.warning("输出不是合法 JSON，追加严格指令后重试一次。")
        del inputs, outputs, generated_ids
        _maybe_empty_cuda_cache()
        strict_instruction = _json_instruction(request.response_format, strict=True)
        retry_messages = [*request.messages, ChatMessage(role="user", content=strict_instruction)]
        outputs, response_text, generated_ids, input_length, inputs = _run_generation(
            request, messages=retry_messages
        )

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
                content_list.append(
                    LogprobContent(token=token_text, logprob=float(score.cpu().item()))
                )
            logprobs_data = ChoiceLogprobs(content=content_list)
        except Exception:  # noqa: BLE001
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
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_length,
                completion_tokens=len(generated_ids),
                total_tokens=input_length + len(generated_ids),
            ),
        )
    finally:
        del inputs, outputs, generated_ids
        _maybe_empty_cuda_cache()


# ── 启动入口 ──


def start_api_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    request_timeout: float = 300,
    mode: str = "full",
    model_path: Optional[str] = None,
    base_model_path: Optional[str] = None,
    lora_path: Optional[str] = None,
    service: Optional[str] = None,
) -> None:
    """启动高并发 OpenAI API 兼容服务。

    参数：
        host, port: 监听地址
        workers: uvicorn worker 数量。多 GPU 场景可 >1，各 worker 独占显存
        request_timeout: 单个请求的最大等待+推理时间（秒）
        mode: 'full' 或 'lora'
        model_path: 模型路径（full mode 或 lora 的适配器路径）
        base_model_path: LoRA 模式下的基座模型路径
        lora_path: LoRA 适配器路径（默认取 model_path）
        service: 服务名（input/output/chat），决定 configs/services.env 的推理参数前缀
    """
    configure_model(
        mode=mode,
        model_path=model_path,
        base_model_path=base_model_path,
        lora_path=lora_path,
        request_timeout=request_timeout,
        service=service,
    )

    # workers=1 时预加载模型（首次请求不等待）
    if workers == 1:
        try:
            _ensure_model_loaded()
        except Exception as exc:  # noqa: BLE001
            logger.warning("模型预加载失败（首次请求时会重试）: %s", exc)

    if workers > 1:
        logger.warning("多 worker 模式 (workers=%s)：每个子进程会独立加载模型。", workers)

    logger.info("高并发推理服务启动: http://%s:%s (workers=%s, timeout=%ss)", host, port, workers, request_timeout)

    if workers > 1:
        uvicorn.run(
            "injection_guard.services.llm_server:app", host=host, port=port, workers=workers
        )
    else:
        uvicorn.run(app, host=host, port=port)
