"""下游模型服务连通性探测。

流水线依赖四个 OpenAI 兼容服务：输入检测 LLM / 输出检测 LLM / Embedding(RAG) / 聊天 LLM。
本模块对每个服务各发一次**最小真实请求**，验证「连得上 + 模型可用」，返回延迟与失败原因，
供 ``POST /api/test_services``（前端设置页「服务连通性」）使用。

只依赖 ``openai`` 与 ``requests`` 层的 ``Settings``，**不 import torch**，
故可在无 GPU 环境下被导入与测试。
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

# 探测目标键（固定顺序：输入 / 输出 / RAG / 聊天）
SERVICE_KEYS = ("input", "output", "rag", "chat")

# 单次探测默认超时（秒）。取值考虑**冷启动**：本地 8B 检测模型空闲后首次推理
# 实测 3s+（CUDA 预热），5s 会误报超时；10s 既容得下冷启动，健康时也在 1s 内返回。
# 调用方仍可用 timeout 放大（上限 60s，足以覆盖真正的模型懒加载）。
DEFAULT_TIMEOUT = 10.0
MIN_TIMEOUT = 0.5
MAX_TIMEOUT = 60.0

# openai SDK 不接受空 key；本地服务忽略该 Authorization 头，
# 故无鉴权的服务统一用占位串（同 chat_client / safety_detector 的既有约定）。
_PLACEHOLDER_KEY = "EMPTY"

# 探测用的最小请求负载
_PROBE_INPUT = "ping"
_PROBE_MAX_TOKENS = 1


@dataclass(frozen=True)
class ProbeTarget:
    """一个待探测的下游服务。"""

    key: str
    label: str
    base_url: str
    model: str
    kind: str  # "chat" | "embedding"
    api_key: str


def normalize_timeout(timeout: Optional[float]) -> float:
    """把请求给的超时归一到 [MIN_TIMEOUT, MAX_TIMEOUT]；None 取默认值。"""
    if timeout is None:
        return DEFAULT_TIMEOUT
    return min(max(float(timeout), MIN_TIMEOUT), MAX_TIMEOUT)


def resolve_targets(settings: Optional[Settings] = None) -> Dict[str, ProbeTarget]:
    """由 ``Settings`` 派生四个探测目标（键 ``SERVICE_KEYS``）。"""
    settings = settings or get_settings()
    return {
        "input": ProbeTarget(
            key="input",
            label="输入检测 LLM",
            base_url=settings.input_detector_api_url,
            model=settings.input_detector_api_model,
            kind="chat",
            api_key=_PLACEHOLDER_KEY,  # 输入检测无 api_key 配置项
        ),
        "output": ProbeTarget(
            key="output",
            label="输出检测 LLM",
            base_url=settings.output_detector_api_url,
            model=settings.output_detector_model,
            kind="chat",
            api_key=settings.output_detector_api_key or _PLACEHOLDER_KEY,
        ),
        "rag": ProbeTarget(
            key="rag",
            label="RAG(Embedding)",
            base_url=settings.embed_api_url,
            model=settings.embed_model_name,
            kind="embedding",
            api_key=_PLACEHOLDER_KEY,  # Embedding 服务无鉴权
        ),
        "chat": ProbeTarget(
            key="chat",
            label="聊天 LLM",
            base_url=settings.chat_api_url,
            model=settings.chat_model,
            kind="chat",
            api_key=settings.chat_api_key or _PLACEHOLDER_KEY,
        ),
    }


def _invoke(target: ProbeTarget, timeout: float) -> None:
    """发一次最小请求；失败由调用方捕获。

    ``max_retries=0``：诊断要的是「当下能否连上」，重试只会拖长失败反馈。
    """
    client = OpenAI(
        api_key=target.api_key, base_url=target.base_url, timeout=timeout, max_retries=0
    )
    if target.kind == "embedding":
        client.embeddings.create(model=target.model, input=_PROBE_INPUT)
    else:
        client.chat.completions.create(
            model=target.model,
            messages=[{"role": "user", "content": _PROBE_INPUT}],
            max_tokens=_PROBE_MAX_TOKENS,
        )


def probe(target: ProbeTarget, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """探测单个服务，返回 ``{ok, label, url, model, latency_ms, error}``。

    任何异常都被转成 ``ok=False`` + ``error`` 文本，绝不冒泡——诊断接口必须始终有结果。
    """
    started = time.perf_counter()
    error: Optional[str] = None
    ok = True
    try:
        _invoke(target, timeout)
    except Exception as exc:  # noqa: BLE001 —— 诊断需要把任何失败都变成结果
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        logger.info("[连通性] %s 失败: %s", target.key, error)

    return {
        "ok": ok,
        "label": target.label,
        "url": target.base_url,
        "model": target.model,
        "latency_ms": round((time.perf_counter() - started) * 1000, 1),
        "error": error,
    }


def check_services(
    settings: Optional[Settings] = None,
    services: Optional[Iterable[str]] = None,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """并发探测选定的服务（缺省四个全测）。

    返回 ``{ok, timeout, services: {key: 探测结果}}``；``ok`` 为**全部**选中服务都通过。
    未知的服务名会被忽略（路由层负责校验并返回 400）。
    """
    effective_timeout = normalize_timeout(timeout)
    targets = resolve_targets(settings)

    if services is None:
        selected: List[ProbeTarget] = [targets[key] for key in SERVICE_KEYS]
    else:
        wanted = {str(name).strip().lower() for name in services}
        selected = [targets[key] for key in SERVICE_KEYS if key in wanted]

    results: Dict[str, Dict[str, Any]] = {}
    if selected:
        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            futures = {
                executor.submit(probe, target, effective_timeout): target for target in selected
            }
            for future, target in futures.items():
                results[target.key] = future.result()

    return {
        "ok": all(item["ok"] for item in results.values()),
        "timeout": effective_timeout,
        "services": results,
    }
