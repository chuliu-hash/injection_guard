"""输入侧检测：BERT 模型组 + LLM 复审。

``Detector`` 管理若干本地 BERT 分类模型，对其恶意概率做置信度加权投票，
并在置信度不足或概率落在模糊区间时调用本地 LLM 做复审。
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..config import SENSITIVITY_LEVELS, get_settings
from ..llm_utils import message_text
from .confidence_vote import confidence_weighted_vote
from .risk_mapping import CATEGORY_NAMES, RISK_LEVEL_MAPPING

logger = logging.getLogger(__name__)

# LLM 复审的单轮用户消息模板
LLM_REVIEW_PROMPT = ChatPromptTemplate.from_messages([("human", "{text}")])

# 敏感度档位与阈值表：可经请求体 ``sensitivity`` **逐次**指定（缺省取 ``Settings.default_sensitivity``）。
# 阈值只在单次调用内以局部变量参与判定，故同一个 Detector 实例可被并发请求复用，
# 不会出现「一个请求改了阈值、另一个请求跟着变」的串扰。
# 键必须与 ``config.SENSITIVITY_LEVELS``（API 层校验用）一致。
SENSITIVITY_CONFIGS: Dict[str, Dict[str, Any]] = {
    "low": {"dec_thresh": 0.97, "llm_conf": 0.7, "llm_p": (0.45, 0.55), "llm_accept": 0.99},
    "medium": {"dec_thresh": 0.5, "llm_conf": 0.7, "llm_p": (0.3, 0.7), "llm_accept": 0.80},
    "high": {"dec_thresh": 0.3, "llm_conf": 0.8, "llm_p": (0.2, 0.8), "llm_accept": 0.50},
}

if tuple(SENSITIVITY_CONFIGS) != SENSITIVITY_LEVELS:  # pragma: no cover - 导入期一致性自检
    raise RuntimeError(
        f"SENSITIVITY_CONFIGS 档位 {tuple(SENSITIVITY_CONFIGS)} 与 "
        f"config.SENSITIVITY_LEVELS {SENSITIVITY_LEVELS} 不一致"
    )


def normalize_sensitivity(level: Optional[str], fallback: str = "low") -> str:
    """把敏感度归一到 ``low`` / ``medium`` / ``high``。

    未知值（非空）打告警日志并回退到 ``fallback``（应已是归一值）。
    """
    normalized = (level or "").strip().lower()
    if normalized not in SENSITIVITY_CONFIGS:
        if normalized:
            logger.warning("[Detector] 未知检测强度 %r，回退到 %r", level, fallback)
        return fallback
    return normalized



# 兜底：部分本地 BERT 目录（如 mdeberta-prompt-injection-detector）只带 SentencePiece
# 词表（``spm.model``）而无 fast 分词器所需的 ``tokenizer.json``；此时默认的 fast
# 加载会因离线转换失败而报错，需退回慢速分词器并显式指定词表文件。
_SENTENCEPIECE_VOCAB_FILES = ("spm.model", "sentencepiece.bpe.model", "tokenizer.model")


def _find_sentencepiece_vocab(path: str) -> Optional[str]:
    """在模型目录内查找 SentencePiece 词表文件，找不到返回 ``None``。"""
    for name in _SENTENCEPIECE_VOCAB_FILES:
        candidate = os.path.join(path, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_bert_tokenizer(path: str):
    """加载本地 BERT 目录的分词器，缺 fast 词表时回退慢速 SentencePiece。

    先按默认（fast）方式加载；失败且目录内存在 SentencePiece 词表时，退回
    ``use_fast=False`` + 显式 ``vocab_file``（需已安装 ``sentencepiece``）。两条路径
    都失败则同时保留原始异常链抛出，由调用方决定是否跳过该模型。
    """
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception as fast_exc:  # noqa: BLE001 - 需按兜底结果重新抛出
        vocab_file = _find_sentencepiece_vocab(path)
        if vocab_file is None:
            raise
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, vocab_file=vocab_file)
        except Exception as slow_exc:  # noqa: BLE001 - 保留 fast 与 slow 两条异常链
            raise fast_exc from slow_exc
        logger.info(
            "[Detector] %s 缺 fast 词表，已回退慢速 SentencePiece 分词器（%s）",
            path,
            os.path.basename(vocab_file),
        )
        return tokenizer


@dataclass
class DetectionResult:
    """结构化的检测结果。"""

    is_safe: bool
    risk_level: str
    category: str
    confidence: float
    p_safe: float  # 安全概率
    p_attack: float  # 潜在攻击/注入概率
    decision_by: str
    raw_details: List[Dict]  # 整个检测链路的详细数据（含每个模型的结果）


class Detector:
    """提示注入防御检测器（第一步：BERT 模型组检测）。

    包含 LLM 回退复审机制，支持 Logprobs 置信度校验。
    """

    def __init__(
        self,
        label_map: Optional[Dict[int, str]] = None,
        use_bert: bool = True,
        enable_llm_fallback: bool = True,
        llm_api_url: Optional[str] = None,
        llm_api_model: Optional[str] = None,
        llm_api_timeout: float = 60.0,
        device: Optional[str] = None,
    ) -> None:
        settings = get_settings()

        # 设备选择
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._infer_lock = threading.Lock()

        # 标签映射
        self.label_map = label_map or {0: "安全", 1: "潜在风险提示词"}
        self.models: List[Dict] = []

        # 核心开关与参数配置
        self.use_bert = use_bert
        self.enable_llm_fallback = enable_llm_fallback

        # LLM 参数（优先使用传入参数，否则取配置项 / 环境变量）
        self.llm_api_url = llm_api_url or settings.input_detector_api_url
        self.llm_api_model = llm_api_model or settings.input_detector_api_model
        self.llm_api_timeout = llm_api_timeout

        # LLM 复审走 LangChain ChatModel（OpenAI 兼容）；logprobs 用于取首 token 置信度，
        # 生成长度由推理服务决定（客户端不传 max_tokens）。本地服务无鉴权 -> 占位 key。
        self._llm_model = ChatOpenAI(
            model=self.llm_api_model,
            base_url=self.llm_api_url,
            api_key="EMPTY",
            timeout=self.llm_api_timeout,
            temperature=0.0,
            top_p=0.9,
            logprobs=True,
            top_logprobs=1,
        )

        # 敏感度基线（启动默认值）；单次调用可经 detect_* 的 sensitivity 参数覆盖
        self.set_sensitivity(settings.default_sensitivity)

    def set_bert(self, model_paths: Optional[List[str]] = None) -> None:
        if not self.use_bert:
            return

        if model_paths is None:
            # 不硬编码子目录名：扫描 BERT 根目录下**所有**模型目录（含 config.json 的子目录），
            # 排序保证顺序稳定。增删模型只需在根目录下增删文件夹，无需改代码。
            root = Path(get_settings().bert_model_dir)
            model_paths = (
                sorted(
                    str(child)
                    for child in root.iterdir()
                    if child.is_dir() and (child / "config.json").is_file()
                )
                if root.is_dir()
                else []
            )
            if not model_paths:
                logger.warning("[Detector] 未在 %s 下发现任何 BERT 模型目录", root)

        self.model_paths = model_paths
        self._load_bert_models()

    def set_sensitivity(self, level: str) -> None:
        """设定本 Detector 的**默认**敏感度档位（单次调用可经 ``detect_*`` 的
        ``sensitivity`` 参数覆盖，见 ``normalize_sensitivity``）。"""
        level = normalize_sensitivity(level, self.__dict__.get("sensitivity", "low"))

        self.sensitivity = level

        cfg = SENSITIVITY_CONFIGS[level]
        self.decision_threshold = cfg["dec_thresh"]
        self.llm_conf_threshold = cfg["llm_conf"]
        self.llm_p_range = cfg["llm_p"]
        self.llm_accept_threshold = cfg["llm_accept"]

        logger.info(
            "[Detector] 默认检测强度: %s (BERT阈值=%s, LLM采纳阈值=%s), use_bert:%s",
            level,
            self.decision_threshold,
            self.llm_accept_threshold,
            self.use_bert,
        )

    def _load_bert_models(self) -> None:
        loaded_count = 0
        for path in self.model_paths:
            if not path or not os.path.exists(path):
                continue
            try:
                tokenizer = load_bert_tokenizer(path)
                model = AutoModelForSequenceClassification.from_pretrained(path)
                model.to(self.device)
                model.eval()

                self.models.append(
                    {
                        "name": path.rstrip("/\\").rsplit("/", 1)[-1],
                        "path": path,
                        "tokenizer": tokenizer,
                        "model": model,
                        "injection_index": self._infer_injection_index(
                            getattr(model, "config", None)
                        ),
                    }
                )
                loaded_count += 1
            except Exception as exc:  # noqa: BLE001 - 单个模型失败不应中断整体加载
                logger.warning("[Detector] 加载失败: %s，错误: %s", path, exc)

        if loaded_count == 0:
            logger.warning("未能成功加载任何 BERT 模型。")

    def detect_with_bert_group(
        self,
        text: str,
        sensitivity: Optional[str] = None,
        use_bert: Optional[bool] = None,
    ) -> List[Dict]:
        """BERT 组检测 + 可选 LLM 复审。

        ``sensitivity`` 为**本次调用**的敏感度档位（``low`` / ``medium`` / ``high``）；
        缺省用本实例的默认档位（``set_sensitivity`` 设定）。阈值仅以局部变量参与
        判定，故并发请求复用同一实例时互不干扰。

        ``use_bert`` 为**本次调用**是否使用已加载的 BERT 分类器组（``None`` = 用实例默认）。
        模型加载与否由 ``Detector(use_bert=...)``（服务端 ``ENABLE_BERT``）决定：服务端未启用时
        即便传 ``True`` 也只走 LLM 复审，不会假装安全。
        """
        level = (
            self.sensitivity
            if sensitivity is None
            else normalize_sensitivity(sensitivity, self.sensitivity)
        )
        use_bert_active = self.use_bert if use_bert is None else bool(use_bert and self.use_bert)
        cfg = SENSITIVITY_CONFIGS[level]
        dec_thresh: float = cfg["dec_thresh"]
        llm_conf_threshold: float = cfg["llm_conf"]
        llm_p_range: Tuple[float, float] = cfg["llm_p"]
        llm_accept_threshold: float = cfg["llm_accept"]

        results: List[Dict] = []

        # 1. BERT 组检测（逐个模型前向；未启用则跳过，直接进入 LLM 复审）
        if not use_bert_active:
            logger.debug("[Detector] 本次调用未启用 BERT 组，直接进入 LLM 复审")
        for entry in self.models if use_bert_active else []:
            start_time = time.time()
            try:
                encoding = entry["tokenizer"](
                    text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
                )
                encoding = {k: v.to(self.device) for k, v in encoding.items()}

                with self._infer_lock:
                    with torch.inference_mode():
                        outputs = entry["model"](**encoding)
                        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
                        pred_idx = int(torch.argmax(outputs.logits, dim=-1).item())

                p_safe_val = float(probabilities[0].item())
                p_attack_val = float(probabilities[1].item())

                results.append(
                    {
                        "model_name": entry["name"],
                        "prediction": pred_idx,
                        "label": self.label_map.get(pred_idx, str(pred_idx)),
                        "confidence_score": float(probabilities[pred_idx].item()),
                        "probabilities": {"安全": p_safe_val, "潜在风险提示词": p_attack_val},
                        "p_safe": p_safe_val,
                        "p_attack": p_attack_val,
                        "response_time": time.time() - start_time,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "model_name": entry["name"],
                        "error": str(exc),
                        "response_time": time.time() - start_time,
                    }
                )

        # 2. 投票与初步决策
        prob_list = [r["p_attack"] for r in results if "p_attack" in r]

        # BERT 组被要求使用却没有任何模型产出概率（模型加载失败 / 显存不足等）：此时的
        # 0.0 概率不代表「安全」，不得据此放行，必须交给 LLM 复审。
        bert_group_unavailable = use_bert_active and not prob_list
        if bert_group_unavailable:
            logger.warning("[Detector] BERT 组无可用模型，本次强制 LLM 复审")
            if not self.enable_llm_fallback:
                results.append(
                    {
                        "model_name": "bert_group",
                        "error": "BERT 分类器组无可用模型（可能加载失败或显存不足）",
                        "response_time": 0.0,
                    }
                )

        if prob_list:
            p_final, confidence_score, diag = confidence_weighted_vote(prob_list)
        else:
            p_final, confidence_score, diag = 0.0, 1.0, {}

        pred_final = 1 if p_final > dec_thresh else 0

        ensemble_p_safe = float(1.0 - p_final)
        ensemble_p_attack = float(p_final)

        ensemble_result = {
            "model_name": "ensemble_confidence_weighted_vote",
            "prediction": pred_final,
            "label": self.label_map.get(pred_final, str(pred_final)),
            "confidence_score": float(p_final if pred_final == 1 else (1.0 - p_final)),
            "probabilities": {"安全": ensemble_p_safe, "潜在风险提示词": ensemble_p_attack},
            "p_safe": ensemble_p_safe,
            "p_attack": ensemble_p_attack,
            "response_time": sum(r.get("response_time", 0.0) for r in results),
            "vote_meta": diag,
            "final_decision_by": "bert_ensemble_only",
        }

        # 3. 决定是否 LLM 复审
        if not self.enable_llm_fallback:
            ensemble_result["decision"] = "不安全" if pred_final == 1 else "安全"
        else:
            low_p, high_p = llm_p_range
            needs_llm_review = (
                confidence_score < llm_conf_threshold or low_p < p_final < high_p
            ) or not use_bert_active or bert_group_unavailable
            ensemble_result["decision"] = (
                "LLM_review"
                if needs_llm_review
                else ("不安全" if p_final > dec_thresh else "安全")
            )

        # 4. 执行 LLM 复审
        if ensemble_result["decision"] == "LLM_review" and self.enable_llm_fallback:
            llm_result = self._invoke_llm_fallback(text, llm_accept_threshold=llm_accept_threshold)

            llm_pred = llm_result.get("prediction", 0)
            llm_conf = float(llm_result.get("confidence_score", 0.0))
            llm_raw_pred = llm_result.get("llm_raw_prediction", llm_pred)

            llm_p_attack = llm_conf if llm_raw_pred == 1 else 1.0 - llm_conf
            llm_p_safe = 1.0 - llm_conf if llm_raw_pred == 1 else llm_conf

            llm_result["p_safe"] = llm_p_safe
            llm_result["p_attack"] = llm_p_attack
            results.append(llm_result)

            ensemble_result.update(
                {
                    "decision": "不安全" if llm_pred == 1 else "安全",
                    "final_decision_by": "llm_fallback",
                    "llm_prediction": llm_pred,
                    "prediction": llm_pred,
                    "label": "潜在风险提示词" if llm_pred == 1 else "安全",
                    "confidence_score": llm_p_attack if llm_pred == 1 else llm_p_safe,
                    "probabilities": {"安全": llm_p_safe, "潜在风险提示词": llm_p_attack},
                    "p_safe": llm_p_safe,
                    "p_attack": llm_p_attack,
                    # 提取风险映射信息（如果存在）同步到最终结果
                    "mapped_risk_level": llm_result.get(
                        "mapped_risk_level", "无风险" if llm_pred == 0 else "未知风险"
                    ),
                    "mapped_category": llm_result.get(
                        "mapped_category", "正常" if llm_pred == 0 else "未知分类"
                    ),
                }
            )

        results.append(ensemble_result)
        return results

    def _invoke_llm_fallback(
        self, text: str, llm_accept_threshold: Optional[float] = None
    ) -> Dict:
        """调用 LLM 进行复审。

        ``llm_accept_threshold`` 为本次调用的采纳阈值（缺省用实例默认档位的值）。
        """
        accept_threshold = (
            self.llm_accept_threshold
            if llm_accept_threshold is None
            else llm_accept_threshold
        )
        start_time = time.time()
        try:
            # 生成长度由推理服务（INPUT_API_SERVER_MAX_NEW_TOKENS）决定，客户端不指定。
            response = self._llm_model.invoke(LLM_REVIEW_PROMPT.format_messages(text=text))
            content = message_text(response.content).strip()

            text_is_unsafe = ("unsafe" in content.lower()) or content.startswith("unsafe")
            llm_raw_prediction = 1 if text_is_unsafe else 0

            # 安全地提取 logprob 置信度（LangChain 把 choices[0].logprobs 放进 response_metadata）
            confidence_score = 1.0
            try:
                logprobs = response.response_metadata.get("logprobs") or {}
                first_token = logprobs.get("content", [{}])[0]
                if first_token and first_token.get("logprob") is not None:
                    confidence_score = math.exp(first_token["logprob"])
            except (IndexError, AttributeError, TypeError):
                pass

            final_is_unsafe = text_is_unsafe and (confidence_score >= accept_threshold)

            code = None
            risk_level_str = "无风险"
            category_name_str = "正常"

            if final_is_unsafe:
                m = re.search(r"\bS\d{1,2}\b", content)
                if m:
                    code = m.group(0)
                    risk_level_str = RISK_LEVEL_MAPPING.get(code, "中风险")
                    category_name_str = CATEGORY_NAMES.get(code, code)
                else:
                    risk_level_str = "中风险"
                    category_name_str = "未知类别攻击"

            if risk_level_str == "低风险":
                final_is_unsafe = False

            return {
                "model_name": "llm_fallback",
                "prediction": 1 if final_is_unsafe else 0,
                "label": "潜在风险提示词" if final_is_unsafe else "安全",
                "response_time": time.time() - start_time,
                "llm_raw_response": content,
                "confidence_score": confidence_score,
                "llm_raw_prediction": llm_raw_prediction,
                "risk_code": code,
                "mapped_risk_level": risk_level_str,
                "mapped_category": category_name_str,
            }

        except Exception as exc:  # noqa: BLE001
            logger.exception("[Detector] LLM 复审异常")
            return {
                "model_name": "llm_fallback",
                "prediction": 0,
                "label": "安全",
                "confidence_score": 0.0,
                "error": str(exc),
                "response_time": time.time() - start_time,
                "llm_raw_prediction": 0,
            }

    def _infer_injection_index(self, config: Any) -> int:
        try:
            id2label = getattr(config, "id2label", {})
            items = id2label.items() if isinstance(id2label, dict) else enumerate(id2label)

            for k, v in items:
                if isinstance(v, str) and any(
                    s in v.lower() for s in ["injection", "unsafe", "attack"]
                ):
                    return int(k)
        except Exception:  # noqa: BLE001
            pass
        return 1

    def batch_detect(self, texts: List[str], sensitivity: Optional[str] = None) -> List[Dict]:
        """批量检测多个文本（真正的批量推理）。

        ``sensitivity`` 语义与 ``detect_with_bert_group`` 相同（缺省用实例默认档位）。
        """
        if not texts:
            return []

        level = (
            self.sensitivity
            if sensitivity is None
            else normalize_sensitivity(sensitivity, self.sensitivity)
        )
        dec_thresh: float = SENSITIVITY_CONFIGS[level]["dec_thresh"]

        results = []

        for entry in self.models:
            start_time = time.time()
            try:
                # 真正的批量 tokenize
                encodings = entry["tokenizer"](
                    texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt"
                )
                encodings = {k: v.to(self.device) for k, v in encodings.items()}

                # 批量推理
                with self._infer_lock:
                    with torch.inference_mode():
                        outputs = entry["model"](**encodings)
                        probabilities = torch.softmax(outputs.logits, dim=-1)
                        pred_indices = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

                # 处理批量结果
                for i, _text in enumerate(texts):
                    p_safe_val = float(probabilities[i][0].item())
                    p_attack_val = float(probabilities[i][1].item())
                    pred_idx = pred_indices[i]

                    results.append(
                        {
                            "model_name": entry["name"],
                            "prediction": pred_idx,
                            "label": self.label_map.get(pred_idx, str(pred_idx)),
                            "confidence_score": float(probabilities[i][pred_idx].item()),
                            "probabilities": {"安全": p_safe_val, "潜在风险提示词": p_attack_val},
                            "p_safe": p_safe_val,
                            "p_attack": p_attack_val,
                            "response_time": time.time() - start_time,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                for _text in texts:
                    results.append(
                        {
                            "model_name": entry["name"],
                            "error": str(exc),
                            "response_time": time.time() - start_time,
                        }
                    )

        # 为每个文本进行投票与决策
        num_texts = len(texts)
        num_models = len(self.models)
        final_results = []

        for text_idx in range(num_texts):
            text_results = results[text_idx * num_models : (text_idx + 1) * num_models]
            prob_list = [r["p_attack"] for r in text_results if "p_attack" in r]

            if prob_list:
                p_final, confidence_score, _diag = confidence_weighted_vote(prob_list)
            else:
                p_final, confidence_score = 0.0, 1.0

            pred_final = 1 if p_final > dec_thresh else 0
            ensemble_p_safe = float(1.0 - p_final)
            ensemble_p_attack = float(p_final)

            final_results.append(
                {
                    "model_name": "ensemble_confidence_weighted_vote",
                    "prediction": pred_final,
                    "label": self.label_map.get(pred_final, str(pred_final)),
                    "confidence_score": float(confidence_score),
                    "probabilities": {"安全": ensemble_p_safe, "潜在风险提示词": ensemble_p_attack},
                    "p_safe": ensemble_p_safe,
                    "p_attack": ensemble_p_attack,
                    "response_time": sum(r.get("response_time", 0) for r in text_results)
                    / max(len(text_results), 1),
                }
            )

        return final_results


def detect_input(
    text: str,
    sensitivity: str = "medium",
    use_bert: bool = False,
    llm_api_url: Optional[str] = None,
    llm_api_model: Optional[str] = None,
    **kwargs: Any,
) -> DetectionResult:
    """便捷的输入提示词检测函数。

    支持灵活配置是否使用 BERT、以及本地或云端大模型的 URL 和名称。
    """
    # 1. 实例化检测器（None 值不传入，使用 Detector 类内的默认值）
    detector_kwargs: Dict[str, Any] = {"use_bert": use_bert}
    if llm_api_url is not None:
        detector_kwargs["llm_api_url"] = llm_api_url
    if llm_api_model is not None:
        detector_kwargs["llm_api_model"] = llm_api_model
    detector_kwargs.update(kwargs)

    detector = Detector(**detector_kwargs)

    # 2. 启动 BERT 模型组
    if use_bert:
        detector.set_bert()

    # 3. 设置检测敏感度及阈值参数
    detector.set_sensitivity(sensitivity)

    # 4. 执行检测
    results = detector.detect_with_bert_group(text)

    final_result = results[-1]
    is_safe = final_result.get("prediction") == 0

    # 提取概率
    probs_dict = final_result.get("probabilities", {})
    p_safe = final_result.get("p_safe", probs_dict.get("安全", 1.0 if is_safe else 0.0))
    p_attack = final_result.get(
        "p_attack", probs_dict.get("潜在风险提示词", 0.0 if is_safe else 1.0)
    )

    # 提取风险映射
    risk_level = final_result.get("mapped_risk_level", "无风险" if is_safe else "未知风险")
    category = final_result.get("mapped_category", "正常" if is_safe else "未知分类")

    return DetectionResult(
        is_safe=is_safe,
        risk_level=risk_level,
        category=category,
        confidence=final_result.get("confidence_score", 0.0),
        p_safe=float(p_safe),
        p_attack=float(p_attack),
        decision_by=final_result.get("final_decision_by", "unknown"),
        raw_details=results,
    )
