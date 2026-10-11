"""防御流水线的服务层与 FastAPI 依赖。

把原 ``app_pj.py`` 中的进程级单例（输入检测器池 / 输出检测器）与并发闸门
收敛到 ``DefenseService``，通过 FastAPI 依赖注入，便于测试替换。
"""

from __future__ import annotations

import itertools
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from ..config import SENSITIVITY_LEVELS, Settings, get_settings
from ..errors import DetectionError, ServiceBusyError  # noqa: F401  (再导出，供 api/测试沿用)
from ..services.gpu import parse_gpu_ids

if TYPE_CHECKING:  # 仅类型检查时导入，避免把 torch/transformers 带到模块加载
    from ..agent.pipeline import DefenseAgent
    from ..defense.prompt_detection import Detector
    from ..defense.rag import SCR_RAG_System
    from ..defense.safety_detector import SafetyDetector
    from ..services.chat_client import ChatClient

logger = logging.getLogger(__name__)

# LangGraph Agent 惰性构建锁（跨实例共享，避免并发重复编译状态图）
_AGENT_BUILD_LOCK = threading.Lock()


@dataclass
class DefendOutcome:
    """防御流水线的结果。"""

    is_safe: bool
    stopped_at: Optional[str]
    input_detection: Optional[Dict[str, Any]]
    output_detection: Optional[Dict[str, Any]]
    model_response: Optional[str] = None
    rag: Optional[Dict[str, Any]] = None


def validate_defend_request(data: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """校验请求体。返回 ``(ok, error_message)``。

    ``user_input`` 必填；``model_response`` 可选（缺省时由流水线调用聊天接口生成），
    但若显式提供则必须是非空字符串。
    """
    if not data:
        return False, "请求体必须是 JSON"

    if "user_input" not in data:
        return False, "缺少必填字段: user_input"
    if not isinstance(data["user_input"], str):
        return False, "user_input 必须是字符串"
    if not data["user_input"].strip():
        return False, "user_input 不能为空"

    model_response = data.get("model_response")
    if model_response is not None:
        if not isinstance(model_response, str):
            return False, "model_response 必须是字符串"
        if not model_response.strip():
            return False, "model_response 不能为空"

    # 聊天生成参数（可选，前端设置面板提供）
    temperature = data.get("temperature")
    if temperature is not None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            return False, "temperature 必须是数字"
        if not 0.0 <= float(temperature) <= 2.0:
            return False, "temperature 必须在 0~2 之间"

    max_tokens = data.get("max_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            return False, "max_tokens 必须是整数"
        if max_tokens < 1:
            return False, "max_tokens 必须 >= 1"

    # 输出检测跳过策略（前端设置面板提供）
    safe_skip_threshold = data.get("safe_skip_threshold")
    if safe_skip_threshold is not None:
        if isinstance(safe_skip_threshold, bool) or not isinstance(
            safe_skip_threshold, (int, float)
        ):
            return False, "safe_skip_threshold 必须是数字"
        if not 0.0 <= float(safe_skip_threshold) <= 1.0:
            return False, "safe_skip_threshold 必须在 0~1 之间"

    skip_flag = data.get("skip_output_when_high_confidence")
    if skip_flag is not None and not isinstance(skip_flag, bool):
        return False, "skip_output_when_high_confidence 必须是布尔值"

    # 输入检测敏感度（前端设置面板提供）
    sensitivity = data.get("sensitivity")
    if sensitivity is not None:
        if not isinstance(sensitivity, str) or sensitivity.strip().lower() not in SENSITIVITY_LEVELS:
            return False, "sensitivity 必须是 low / medium / high 之一"

    return True, None


class DefenseService:
    """持有检测器单例与并发闸门，执行「输入检测 -> 输出检测」流水线。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        import torch

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.input_gpu_ids = parse_gpu_ids(settings.input_detector_gpu_ids, cuda_count)
        self.replicas = max(1, settings.input_detector_replicas_per_gpu)

        default_concurrency = (
            max(1, len(self.input_gpu_ids) * self.replicas) if self.input_gpu_ids else 8
        )
        self.input_max_concurrency = settings.input_max_concurrency or default_concurrency
        self.output_max_concurrency = settings.output_max_concurrency or default_concurrency
        self.chat_max_concurrency = settings.chat_max_concurrency or default_concurrency

        # 并发闸门（单进程内）
        self._input_sem = threading.BoundedSemaphore(self.input_max_concurrency)
        self._output_sem = threading.BoundedSemaphore(self.output_max_concurrency)
        self._chat_sem = threading.BoundedSemaphore(self.chat_max_concurrency)

        # 输入检测器：按 GPU 维护一个小池，避免高并发集中到单个 cuda:0
        self._input_detectors: List[Detector] = []
        self._input_cycle: Optional[itertools.cycle] = None
        self._input_lock = threading.Lock()

        # 输出检测器：全局只缓存一份
        self._output_detector: Optional[SafetyDetector] = None
        self._output_lock = threading.Lock()

        # 聊天客户端 / RAG 系统：全局各缓存一份
        self._chat_client: Optional[ChatClient] = None
        self._chat_lock = threading.Lock()
        self._rag_system: Optional[SCR_RAG_System] = None
        self._rag_lock = threading.Lock()

        # 预编译 LangGraph 状态图（一次编译、进程内复用），把 ~1s 的编译耗时
        # 从「首个请求」挪到启动阶段。
        self._get_agent()

    @property
    def input_detector_devices(self) -> List[str]:
        """已就绪的输入检测器所在设备（用于健康检查）。"""
        return [str(detector.device) for detector in self._input_detectors]

    # ── 检测器构建 / 获取 ────────────────────────────────────────────

    def _build_input_detector(
        self,
        use_bert: bool = False,
        llm_api_url: Optional[str] = None,
        llm_api_model: Optional[str] = None,
        device: Optional[str] = None,
    ) -> Detector:
        kwargs: Dict[str, Any] = {"use_bert": use_bert}
        if llm_api_url is not None:
            kwargs["llm_api_url"] = llm_api_url
        if llm_api_model is not None:
            kwargs["llm_api_model"] = llm_api_model
        if device is not None:
            kwargs["device"] = device

        from ..defense.prompt_detection import Detector

        detector = Detector(**kwargs)
        detector.set_bert()
        detector.set_sensitivity(self.settings.default_sensitivity)
        return detector

    def _get_input_detector(
        self,
        use_bert: bool = False,
        llm_api_url: Optional[str] = None,
        llm_api_model: Optional[str] = None,
    ) -> Detector:
        """获取常驻输入 Detector（轮询池）。

        注意：``use_bert`` / ``llm_api_url`` / ``llm_api_model`` 只在首次初始化时
        生效，后续请求复用同一实例。
        """
        if self._input_detectors:
            return next(self._input_cycle)

        with self._input_lock:
            if self._input_detectors:
                return next(self._input_cycle)

            devices = (
                [f"cuda:{gpu_id}" for gpu_id in self.input_gpu_ids] if self.input_gpu_ids else [None]
            )
            detectors: List[Detector] = []
            for device in devices:
                for _ in range(self.replicas):
                    detectors.append(
                        self._build_input_detector(
                            use_bert=use_bert,
                            llm_api_url=llm_api_url,
                            llm_api_model=llm_api_model,
                            device=device,
                        )
                    )

            self._input_detectors = detectors
            self._input_cycle = itertools.cycle(self._input_detectors)
            logger.info(
                "[预热] 输入检测器池就绪: sensitivity=%s, devices=%s",
                self.settings.default_sensitivity,
                [str(d.device) for d in self._input_detectors],
            )
            return next(self._input_cycle)

    def _get_output_detector(
        self,
        constitution_file: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> SafetyDetector:
        """获取常驻输出 SafetyDetector（全局一份，仅首次初始化参数生效）。"""
        if self._output_detector is not None:
            return self._output_detector

        with self._output_lock:
            if self._output_detector is not None:
                return self._output_detector

            kwargs: Dict[str, Any] = {}
            if constitution_file is not None:
                kwargs["constitution_file"] = constitution_file
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
            if model_name is not None:
                kwargs["model_name"] = model_name

            from ..defense.safety_detector import SafetyDetector

            self._output_detector = SafetyDetector(**kwargs)
            logger.info("[预热] 输出检测器就绪")
            return self._output_detector

    def _get_chat_client(self) -> "ChatClient":
        """获取常驻聊天客户端（全局一份）。"""
        if self._chat_client is not None:
            return self._chat_client

        with self._chat_lock:
            if self._chat_client is None:
                from ..services.chat_client import ChatClient

                self._chat_client = ChatClient()
                logger.info("[预热] 聊天客户端就绪: base_url=%s", self._chat_client.base_url)
            return self._chat_client

    def _get_rag_system(self) -> "SCR_RAG_System":
        """获取常驻 RAG 系统（全局一份，首次使用时加载知识库）。"""
        if self._rag_system is not None:
            return self._rag_system

        with self._rag_lock:
            if self._rag_system is None:
                from ..defense.rag import SCR_RAG_System

                self._rag_system = SCR_RAG_System()
                logger.info("[预热] RAG 系统就绪: top_k=%s", self.settings.rag_top_k)
            return self._rag_system

    # ── 单次检测 ─────────────────────────────────────────────────────

    def _run_input_detection(
        self,
        text: str,
        use_bert: bool = False,
        llm_api_url: Optional[str] = None,
        llm_api_model: Optional[str] = None,
        sensitivity: Optional[str] = None,
    ) -> Dict[str, Any]:
        detector = self._get_input_detector(
            use_bert=use_bert, llm_api_url=llm_api_url, llm_api_model=llm_api_model
        )

        # 敏感度以**局部参数**传入：同一 Detector 实例被并发请求轮询复用时，
        # 阈值只在本次调用内生效，不会串扰其他请求（见 prompt_detection.normalize_sensitivity）。
        results = detector.detect_with_bert_group(text, sensitivity=sensitivity)
        if not results:
            raise RuntimeError("输入检测器返回空结果")

        final_result = results[-1]
        if not isinstance(final_result, dict):
            raise RuntimeError("输入检测器最终结果格式异常")

        is_safe = final_result.get("prediction") == 0

        detection_error = None
        for item in results:
            if isinstance(item, dict) and item.get("error"):
                detection_error = str(item["error"])
                break

        probs_dict = final_result.get("probabilities", {})

        # 回显本次实际生效的档位（未指定时即该 Detector 的默认值）。
        from ..defense.prompt_detection import normalize_sensitivity

        effective_sensitivity = normalize_sensitivity(sensitivity, detector.sensitivity)

        return {
            "is_safe": is_safe,
            "risk_level": final_result.get("mapped_risk_level", "无风险" if is_safe else "未知风险"),
            "category": final_result.get("mapped_category", "正常" if is_safe else "未知分类"),
            "confidence": final_result.get("confidence_score", 0.0),
            "p_safe": final_result.get("p_safe", probs_dict.get("安全", 1.0 if is_safe else 0.0)),
            "p_attack": final_result.get(
                "p_attack", probs_dict.get("潜在风险提示词", 0.0 if is_safe else 1.0)
            ),
            "decision_by": final_result.get("final_decision_by", "unknown"),
            "detection_error": detection_error,
            "sensitivity": effective_sensitivity,
        }

    def _run_input_detection_guarded(
        self,
        text: str,
        use_bert: bool = False,
        llm_api_url: Optional[str] = None,
        llm_api_model: Optional[str] = None,
        sensitivity: Optional[str] = None,
    ) -> Dict[str, Any]:
        """带并发控制的输入检测；获取名额超时抛 ``ServiceBusyError``。"""
        acquired = self._input_sem.acquire(timeout=self.settings.input_acquire_timeout)
        if not acquired:
            raise ServiceBusyError("输入检测服务繁忙，请稍后重试")
        try:
            return self._run_input_detection(
                text=text,
                use_bert=use_bert,
                llm_api_url=llm_api_url,
                llm_api_model=llm_api_model,
                sensitivity=sensitivity,
            )
        finally:
            self._input_sem.release()

    def _run_output_detection(
        self,
        text: str,
        answer: str,
        constitution_file: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Any:
        detector = self._get_output_detector(
            constitution_file=constitution_file,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
        return detector.detect(text=text, answer=answer)

    def _run_output_detection_guarded(
        self,
        text: str,
        answer: str,
        constitution_file: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Any:
        """带并发控制的输出检测；获取名额超时抛 ``ServiceBusyError``。"""
        acquired = self._output_sem.acquire(timeout=self.settings.output_acquire_timeout)
        if not acquired:
            raise ServiceBusyError("输出检测服务繁忙，请稍后重试")
        try:
            return self._run_output_detection(
                text=text,
                answer=answer,
                constitution_file=constitution_file,
                api_key=api_key,
                base_url=base_url,
                model_name=model_name,
            )
        finally:
            self._output_sem.release()

    # ── RAG 增强 / 聊天生成 ──────────────────────────────────────────

    def _run_rag(self, user_input: str) -> Dict[str, Any]:
        """检索相似安全上下文并构建 ``final_prompt``。

        失败时抛异常，由 ``detect`` 降级处理（不阻断主流程）。
        """
        rag_system = self._get_rag_system()
        contexts = rag_system.retrieve(user_input, top_k=self.settings.rag_top_k)
        final_prompt = rag_system.construct_safe_prompt(
            user_query=user_input, pre_retrieved_contexts=contexts
        )
        return {
            "enabled": True,
            "final_prompt": final_prompt,
            # 剔除向量（tensor），只保留可序列化的文本字段
            "retrieved_contexts": [
                {"prompt": ctx.get("prompt", ""), "response": ctx.get("response", "")}
                for ctx in contexts
            ],
        }

    def _run_chat(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """调用普通聊天接口生成回复（生成参数来自请求，留空用 Settings 默认值）。"""
        return self._get_chat_client().complete(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

    def _run_chat_guarded(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """带并发控制的聊天生成；获取名额超时抛 ``ServiceBusyError``。"""
        acquired = self._chat_sem.acquire(timeout=self.settings.chat_acquire_timeout)
        if not acquired:
            raise ServiceBusyError("聊天服务繁忙，请稍后重试")
        try:
            return self._run_chat(prompt, max_tokens=max_tokens, temperature=temperature)
        finally:
            self._chat_sem.release()

    def _run_chat_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """流式调用普通聊天接口，逐段 yield 文本增量。"""
        yield from self._get_chat_client().stream(
            prompt, max_tokens=max_tokens, temperature=temperature
        )

    def _run_chat_stream_guarded(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        """带并发控制的流式聊天生成；获取名额超时抛 ``ServiceBusyError``。

        生成器被提前关闭（客户端断开）时，``finally`` 仍会释放并发名额。
        """
        acquired = self._chat_sem.acquire(timeout=self.settings.chat_acquire_timeout)
        if not acquired:
            raise ServiceBusyError("聊天服务繁忙，请稍后重试")
        try:
            yield from self._run_chat_stream(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
        finally:
            self._chat_sem.release()

    # ── 流水线 ───────────────────────────────────────────────────────

    def _resolve_skip_policy(
        self,
        skip_output_when_high_confidence: Optional[bool],
        safe_skip_threshold: Optional[float],
    ) -> Tuple[bool, float]:
        """输出检测跳过策略：请求值优先，缺省回落 ``Settings``（启动默认值）。"""
        enabled = (
            self.settings.skip_output_when_high_confidence
            if skip_output_when_high_confidence is None
            else skip_output_when_high_confidence
        )
        threshold = (
            self.settings.safe_skip_threshold
            if safe_skip_threshold is None
            else safe_skip_threshold
        )
        return enabled, threshold

    def _get_agent(self) -> "DefenseAgent":
        """惰性构建并缓存 LangGraph Agent。

        用 ``getattr`` 读取缓存：测试里的 ``_StubService`` 不调用 ``__init__``，
        首次调用时按需构建，节点仍会走子类覆盖的 ``_run_*_guarded``。
        """
        agent = getattr(self, "_agent", None)
        if agent is None:
            with _AGENT_BUILD_LOCK:
                agent = getattr(self, "_agent", None)
                if agent is None:
                    from ..agent.pipeline import DefenseAgent

                    agent = DefenseAgent(self)
                    self._agent = agent
        return agent

    def _build_pipeline_state(self, *, stream: bool, **kwargs: Any) -> Dict[str, Any]:
        """把 ``detect`` / ``stream_detect`` 的入参整理成 LangGraph 状态。"""
        skip_hc, skip_threshold = self._resolve_skip_policy(
            kwargs.get("skip_output_when_high_confidence"), kwargs.get("safe_skip_threshold")
        )
        use_rag = kwargs.get("use_rag")
        return {
            "user_input": kwargs["user_input"],
            "model_response": kwargs.get("model_response"),
            # 请求体 > 启动默认值（语义同 RAG 开关）
            "use_rag": self.settings.enable_rag if use_rag is None else use_rag,
            "sensitivity": kwargs.get("sensitivity"),
            "use_bert": kwargs.get("use_bert", False),
            "input_llm_api_url": kwargs.get("input_llm_api_url"),
            "input_llm_api_model": kwargs.get("input_llm_api_model"),
            "constitution_file": kwargs.get("constitution_file"),
            "detector_api_key": kwargs.get("detector_api_key"),
            "detector_api_url": kwargs.get("detector_api_url"),
            "detector_model_name": kwargs.get("detector_model_name"),
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
            "skip_output_when_high_confidence": skip_hc,
            "safe_skip_threshold": skip_threshold,
            "stream": stream,
        }

    def detect(
        self,
        *,
        user_input: str,
        model_response: Optional[str] = None,
        use_rag: Optional[bool] = None,
        sensitivity: Optional[str] = None,
        use_bert: bool = False,
        input_llm_api_url: Optional[str] = None,
        input_llm_api_model: Optional[str] = None,
        constitution_file: Optional[str] = None,
        detector_api_key: Optional[str] = None,
        detector_api_url: Optional[str] = None,
        detector_model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        safe_skip_threshold: Optional[float] = None,
        skip_output_when_high_confidence: Optional[bool] = None,
    ) -> DefendOutcome:
        """执行端到端流水线（由 LangGraph 状态图编排，见 ``agent/pipeline.py``）。

        输入检测 ->（可选 RAG 增强）-> 聊天生成（请求未带 ``model_response`` 时）->
        输出检测。

        - 输入不安全：``stopped_at="input_detection"``；
        - 输入安全且高置信：``stopped_at="output_skipped"``，跳过输出检测（仍返回 model_response）；
        - 走完输出检测且判定不安全：``stopped_at="output_detection"``；
        - 走完输出检测且判定安全：``stopped_at=None``。

        ``model_response`` 一律**原样返回**（不替换成拦截文案）：调用方传了
        ``model_response`` 时那是"待检测的文本"，替换会破坏纯检测用法；流式路径下
        文本也已发出、无从撤回。是否拦截展示由前端按 ``stopped_at`` 决定。
        """
        state = self._build_pipeline_state(
            stream=False,
            user_input=user_input,
            model_response=model_response,
            use_rag=use_rag,
            sensitivity=sensitivity,
            use_bert=use_bert,
            input_llm_api_url=input_llm_api_url,
            input_llm_api_model=input_llm_api_model,
            constitution_file=constitution_file,
            detector_api_key=detector_api_key,
            detector_api_url=detector_api_url,
            detector_model_name=detector_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_skip_threshold=safe_skip_threshold,
            skip_output_when_high_confidence=skip_output_when_high_confidence,
        )
        try:
            final = self._get_agent().run(state)
        except ServiceBusyError:
            logger.warning("[503] 流水线并发名额获取超时")
            raise

        return DefendOutcome(
            is_safe=bool(final.get("is_safe", False)),
            stopped_at=final.get("stopped_at"),
            input_detection=final.get("input_detection"),
            output_detection=final.get("output_detection"),
            model_response=final.get("model_response"),
            rag=final.get("rag"),
        )

    def stream_detect(
        self,
        *,
        user_input: str,
        model_response: Optional[str] = None,
        use_rag: Optional[bool] = None,
        sensitivity: Optional[str] = None,
        use_bert: bool = False,
        input_llm_api_url: Optional[str] = None,
        input_llm_api_model: Optional[str] = None,
        constitution_file: Optional[str] = None,
        detector_api_key: Optional[str] = None,
        detector_api_url: Optional[str] = None,
        detector_model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        safe_skip_threshold: Optional[float] = None,
        skip_output_when_high_confidence: Optional[bool] = None,
    ) -> Iterator[Tuple[str, Any]]:
        """``detect`` 的流式版本，逐事件 yield ``(event, data)``（LangGraph 编排）。

        事件序列：``input_detection`` -> ``rag`` -> ``delta``* -> ``output_detection``
        -> ``done``（提前终止时省略中间事件，直接给 ``done``）。分支、拦截、降级、
        高置信跳过等语义与 ``detect`` 完全一致；异常类型也一致
        （``ServiceBusyError`` / ``DetectionError``），由上层转成 SSE ``error`` 事件。

        输出检测判不安全时 ``done.stopped_at="output_detection"``：此时 ``delta`` 已全部
        发出，**服务端无法撤回**，故由前端在收到 ``output_detection`` 事件后立即用拦截
        文案替换气泡内容（这就是"先流式、结果出来再替换"的实现方式）。
        """
        state = self._build_pipeline_state(
            stream=True,
            user_input=user_input,
            model_response=model_response,
            use_rag=use_rag,
            sensitivity=sensitivity,
            use_bert=use_bert,
            input_llm_api_url=input_llm_api_url,
            input_llm_api_model=input_llm_api_model,
            constitution_file=constitution_file,
            detector_api_key=detector_api_key,
            detector_api_url=detector_api_url,
            detector_model_name=detector_model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            safe_skip_threshold=safe_skip_threshold,
            skip_output_when_high_confidence=skip_output_when_high_confidence,
        )

        cancel_event = threading.Event()
        events = iter(self._get_agent().stream(state, cancel_event=cancel_event))
        try:
            while True:
                try:
                    event = next(events)
                except StopIteration:
                    break
                yield event
        except ServiceBusyError:
            logger.warning("[503] 流水线并发名额获取超时")
            raise
        finally:
            # 客户端断流：先置位取消信号（生成节点在下一个增量处提前结束），再关闭
            # 状态图生成器，避免继续做输出检测、并尽快释放聊天并发名额。
            cancel_event.set()
            close = getattr(events, "close", None)
            if close is not None:
                try:
                    close()
                except Exception:  # noqa: BLE001
                    logger.debug("关闭流式状态图时出现异常", exc_info=True)


# ── FastAPI 依赖 ─────────────────────────────────────────────────────
_service: Optional[DefenseService] = None
_service_lock = threading.Lock()


def get_defense_service() -> DefenseService:
    """返回进程级单例 ``DefenseService``（测试可用 ``app.dependency_overrides`` 覆盖）。"""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = DefenseService(get_settings())
    return _service
