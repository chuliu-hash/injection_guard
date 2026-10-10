"""防御 API 的路由。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from .dependencies import (
    DefenseService,
    DetectionError,
    ServiceBusyError,
    get_defense_service,
    validate_defend_request,
)
from .schemas import DefendRequest, DefendResponse, TestServicesRequest, TestServicesResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# SSE 响应头与心跳间隔
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}
SSE_HEARTBEAT_SECONDS = 15.0


@router.get("/health", summary="健康检查")
def health(service: DefenseService = Depends(get_defense_service)) -> dict:
    """健康检查。

    只表示进程存活与配置值，不代表检测器一定已预热完成。
    """
    settings = service.settings
    return {
        "status": "ok",
        "default_sensitivity": settings.default_sensitivity,
        "input_max_concurrency": service.input_max_concurrency,
        "output_max_concurrency": service.output_max_concurrency,
        "chat_max_concurrency": service.chat_max_concurrency,
        "input_gpu_ids": service.input_gpu_ids,
        "input_detector_replicas_per_gpu": service.replicas,
        "input_detector_devices": service.input_detector_devices,
        "skip_output_when_high_confidence": settings.skip_output_when_high_confidence,
        "safe_skip_threshold": settings.safe_skip_threshold,
        "enable_rag": settings.enable_rag,
    }


@router.post(
    "/api/test_services",
    response_model=TestServicesResponse,
    summary="连通性测试：输入/输出/RAG/聊天 四个下游模型服务",
)
def test_services(
    payload: Optional[TestServicesRequest] = None,
    service: DefenseService = Depends(get_defense_service),
):
    """对四个下游服务各发一次最小请求，验证「能连上 + 模型可用」。

    请求体可省略（默认全测）；可只测其中几个：

    ```json
    {"services": ["input", "chat"], "timeout": 5}
    ```

    服务键 ``input`` / ``output`` / ``rag`` / ``chat``；``timeout`` 为单次探测超时（秒，
    0.5~60，默认 5）。**总是返回 200**，逐个服务成败在 ``services`` 里；仅当参数非法
    （未知服务名 / 空列表 / timeout 越界）时返回 ``400 {"error": ...}``。

    探测不经过流水线并发闸门：即使服务繁忙也要能报出真实状态。
    """
    # 惰性导入：保持 API 层在无重依赖环境下可导入（同 dependencies.py 的既有约定）
    from ..services.connectivity import MAX_TIMEOUT, MIN_TIMEOUT, SERVICE_KEYS, check_services

    data = payload.model_dump(exclude_unset=True) if payload is not None else {}

    services = data.get("services")
    if services is not None:
        if not services:
            return JSONResponse(status_code=400, content={"error": "services 不能为空列表"})
        unknown = [name for name in services if name.strip().lower() not in SERVICE_KEYS]
        if unknown:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"未知服务: {', '.join(unknown)}（可选 {' / '.join(SERVICE_KEYS)}）"
                },
            )

    timeout = data.get("timeout")
    if timeout is not None and not MIN_TIMEOUT <= float(timeout) <= MAX_TIMEOUT:
        return JSONResponse(
            status_code=400,
            content={"error": f"timeout 必须在 {MIN_TIMEOUT}~{MAX_TIMEOUT} 秒之间"},
        )

    return check_services(service.settings, services=services, timeout=timeout)


@router.post(
    "/api/prompt_defend",
    response_model=DefendResponse,
    summary="防御流水线：输入检测 ->（可选 RAG）-> 聊天生成 -> 输出检测",
)
def prompt_defend(
    payload: Optional[DefendRequest] = None,
    service: DefenseService = Depends(get_defense_service),
):
    """端到端防御流水线。

    1. 输入不安全 -> 拦截，``stopped_at="input_detection"``（输入检测强度由请求体
       ``sensitivity`` 决定，缺省取启动默认值 ``DEFAULT_SENSITIVITY``）；
    2. 可选 RAG 增强（``use_rag`` / ``ENABLE_RAG``）-> ``final_prompt``；
    3. 请求未带 ``model_response`` 时，调聊天接口生成回复；
    4. 输入安全且高置信 -> 跳过输出检测，``stopped_at="output_skipped"``；
    5. 否则执行输出检测，``stopped_at=null``。
    """
    # exclude_unset：只保留请求实际提供的字段，以复现旧版「缺少必填字段」消息
    data = payload.model_dump(exclude_unset=True) if payload is not None else None
    ok, error_message = validate_defend_request(data)
    if not ok:
        return JSONResponse(status_code=400, content={"error": error_message})

    try:
        outcome = service.detect(
            user_input=data["user_input"],
            model_response=data.get("model_response"),
            use_rag=payload.use_rag,
            sensitivity=payload.sensitivity,
            use_bert=bool(payload.use_bert),
            input_llm_api_url=payload.input_llm_api_url,
            input_llm_api_model=payload.input_llm_api_model,
            constitution_file=payload.constitution_file,
            detector_api_key=payload.detector_api_key,
            detector_api_url=payload.detector_api_url,
            detector_model_name=payload.detector_model_name,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
            safe_skip_threshold=payload.safe_skip_threshold,
            skip_output_when_high_confidence=payload.skip_output_when_high_confidence,
        )
    except ServiceBusyError:
        return JSONResponse(status_code=503, content={"error": "service_busy"})
    except DetectionError as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return {
        "is_safe": outcome.is_safe,
        "stopped_at": outcome.stopped_at,
        "model_response": outcome.model_response,
        "input_detection": outcome.input_detection,
        "rag": outcome.rag,
        "output_detection": outcome.output_detection,
    }


def _sse_frame(event: str, data: Any) -> str:
    """格式化一帧 SSE。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _sse_event_stream(service: DefenseService, kwargs: Dict[str, Any]):
    """把阻塞的 ``stream_detect`` 生成器桥接成 SSE 事件流。

    ``stream_detect`` 在 worker 线程中执行（内部含阻塞式并发闸门），事件经
    ``asyncio.Queue`` 回传，避免阻塞事件循环；长时间无事件时发送心跳注释，
    防止中间层因空闲断开。
    """
    loop = asyncio.get_running_loop()
    queue: "asyncio.Queue" = asyncio.Queue()
    sentinel = object()
    cancelled = threading.Event()

    def _push(item: Any) -> None:
        try:
            loop.call_soon_threadsafe(queue.put_nowait, item)
        except RuntimeError:
            pass  # 事件循环已关闭（客户端断开）

    def _worker() -> None:
        gen = service.stream_detect(**kwargs)
        try:
            for event, data in gen:
                if cancelled.is_set():
                    break
                _push((event, data))
        except ServiceBusyError:
            _push(("error", {"error": "service_busy"}))
        except DetectionError as exc:
            _push(("error", {"error": str(exc)}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("流式检测异常")
            _push(("error", {"error": str(exc)}))
        finally:
            try:
                gen.close()  # 提前 break 时释放并发闸门
            except Exception:  # noqa: BLE001
                pass
            _push(sentinel)

    thread = threading.Thread(target=_worker, name="defense_stream", daemon=True)
    thread.start()
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=SSE_HEARTBEAT_SECONDS)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is sentinel:
                break
            event, data = item
            yield _sse_frame(event, data)
    finally:
        # 客户端断开：通知 worker 在下一个事件处停止
        cancelled.set()


@router.post(
    "/api/chat_stream",
    summary="流式防御流水线（SSE）：输入检测 ->（可选 RAG）-> 流式聊天 -> 输出检测",
)
async def chat_stream(
    payload: Optional[DefendRequest] = None,
    service: DefenseService = Depends(get_defense_service),
):
    """以 Server-Sent Events 逐步推送流水线事件。

    事件类型：``input_detection`` / ``rag`` / ``delta`` / ``output_detection`` /
    ``done`` / ``error``。请求体校验不通过时返回普通 ``400 {"error": ...}``
    （非流式）。
    """
    data = payload.model_dump(exclude_unset=True) if payload is not None else None
    ok, error_message = validate_defend_request(data)
    if not ok:
        return JSONResponse(status_code=400, content={"error": error_message})

    kwargs: Dict[str, Any] = {
        "user_input": data["user_input"],
        "model_response": data.get("model_response"),
        "use_rag": payload.use_rag,
        "sensitivity": payload.sensitivity,
        "use_bert": bool(payload.use_bert),
        "input_llm_api_url": payload.input_llm_api_url,
        "input_llm_api_model": payload.input_llm_api_model,
        "constitution_file": payload.constitution_file,
        "detector_api_key": payload.detector_api_key,
        "detector_api_url": payload.detector_api_url,
        "detector_model_name": payload.detector_model_name,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "safe_skip_threshold": payload.safe_skip_threshold,
        "skip_output_when_high_confidence": payload.skip_output_when_high_confidence,
    }
    return StreamingResponse(
        _sse_event_stream(service, kwargs),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
