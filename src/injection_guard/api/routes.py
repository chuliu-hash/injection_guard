"""防御 API 的路由。"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse, StreamingResponse

from ..agent.state import final_assistant_content
from ..config import get_settings
from ..persistence.database import get_database
from ..persistence.repository import SessionRepository
from ..services import llm_metrics
from .dependencies import (
    DefenseService,
    DetectionError,
    ServiceBusyError,
    get_defense_service,
    validate_defend_request,
)
from .schemas import DefendRequest, DefendResponse, TestServicesRequest, TestServicesResponse
from .security import get_current_user_id, require_user

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
        # 进程内 LLM 指标（调用次数 / 平均耗时 / token 用量；重启归零）
        "llm_metrics": llm_metrics.snapshot(),
    }


@router.post(
    "/api/test_services",
    response_model=TestServicesResponse,
    summary="连通性测试：输入/输出/RAG/聊天 四个下游模型服务",
    dependencies=[Depends(require_user)],
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
    dependencies=[Depends(require_user)],
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
            use_bert=payload.use_bert,
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


@dataclass(frozen=True)
class _AssistantPersistTarget:
    """流式生成结束后要落库的助手消息定位（请求体同时带会话与消息 id 时才有）。"""

    user_id: str
    session_id: str
    message_id: str

    def persist(self, state: Dict[str, Any]) -> None:
        """把终态正文写回会话；失败只告警，不影响已经完成的流。

        前端也会 PATCH 一次同样的内容（幂等）：以服务端落库为准，保证客户端收到
        ``done`` 后立刻刷新 / 换设备也能看到回复正文。
        """
        content = final_assistant_content(
            model_response=state.get("model_response"), stopped_at=state.get("stopped_at")
        )
        try:
            with get_database().session() as db:
                SessionRepository(db, user_id=self.user_id).patch_message(
                    self.session_id, self.message_id, {"content": content, "status": "done"}
                )
        except Exception:  # noqa: BLE001 - 落库失败不影响已完成的生成
            logger.warning(
                "[stream] 助手消息落库失败: session=%s message=%s",
                self.session_id,
                self.message_id,
                exc_info=True,
            )


async def _sse_event_stream(
    service: DefenseService,
    kwargs: Dict[str, Any],
    persist_target: Optional[_AssistantPersistTarget] = None,
):
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
        done_state: Optional[Dict[str, Any]] = None
        try:
            for event, data in gen:
                # 先记下 done 再判断断流：生成已完成时即使用户已关掉页面，也要落库
                if event == "done" and isinstance(data, dict):
                    done_state = data
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
            if persist_target is not None and done_state is not None:
                persist_target.persist(done_state)
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


def _load_chat_history(
    session_id: Optional[str], user_id: str, user_input: str
) -> Optional[List[Dict[str, str]]]:
    """按会话读取多轮历史（**服务端读取**，前端只传 session_id）。

    只取已完成的消息、剔除本次这条用户输入，并按 ``CHAT_HISTORY_MAX_TURNS`` 截断
    （token 预算的裁剪在 ``ChatClient`` 里做）。任何异常都降级为「无历史」——记忆
    功能不该让聊天失败。
    """
    if not session_id:
        return None
    max_turns = get_settings().chat_history_max_turns
    if max_turns <= 0:
        return None
    try:
        with get_database().session() as db:
            history = SessionRepository(db, user_id=user_id).load_history(
                session_id, max_turns=max_turns, exclude_trailing_user=user_input
            )
    except Exception:  # noqa: BLE001 - 读历史失败按无历史处理
        logger.warning("[chat] 读取会话历史失败: session=%s", session_id, exc_info=True)
        return None
    if history:
        logger.info("[chat] 注入 %d 条历史消息: session=%s", len(history), session_id)
    return history or None


@router.post(
    "/api/chat_stream",
    summary="流式防御流水线（SSE）：输入检测 ->（可选 RAG）-> 流式聊天 -> 输出检测",
    dependencies=[Depends(require_user)],
)
async def chat_stream(
    payload: Optional[DefendRequest] = None,
    service: DefenseService = Depends(get_defense_service),
    user_id: str = Depends(get_current_user_id),
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
        "use_bert": payload.use_bert,
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
        # 多轮记忆：服务端从会话里读历史（前端只传 session_id，不信任前端传的历史）
        "history": _load_chat_history(payload.session_id, user_id, data["user_input"]),
    }
    # 带会话定位时，由服务端在生成结束时落库助手正文（见 _AssistantPersistTarget）
    persist_target: Optional[_AssistantPersistTarget] = None
    if payload.session_id and payload.assistant_message_id:
        persist_target = _AssistantPersistTarget(
            user_id=user_id,
            session_id=payload.session_id,
            message_id=payload.assistant_message_id,
        )

    return StreamingResponse(
        _sse_event_stream(service, kwargs, persist_target),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
