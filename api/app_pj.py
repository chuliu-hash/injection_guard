"""
Flask 后端 API —— AI 安全防御系统
防御流水线：输入检测 → 输出检测

启动方式:
    python app.py

生产建议（高并发）:
    # Flask 内置服务器适合开发和低并发测试
    python app.py

    # 生产环境推荐 gunicorn + gevent（异步 worker）
    gunicorn app:app \
      --bind 0.0.0.0:8099 \
      --worker-class gevent \
      --workers 2 \
      --worker-connections 64 \
      --timeout 120 \
      --backlog 128

    # 如果 gevent 不可用，使用 threaded worker:
    # gunicorn app:app --bind 0.0.0.0:8099 --workers 1 --threads 16 --timeout 120

    高并发说明:
    - input/output 检测各自有 BoundedSemaphore 控制并发数
    - 繁忙时自动降级返回 is_safe=True，不阻塞客户端
    - gunicorn + gevent 可在同一进程内处理大量并发 HTTP 连接

环境变量:
    FLASK_PORT=8099
    FLASK_DEBUG=false

    INPUT_MAX_CONCURRENCY=8              # 输入检测并发上限（默认 8）
    OUTPUT_MAX_CONCURRENCY=4             # 输出检测并发上限（默认 4）

    INPUT_ACQUIRE_TIMEOUT=5              # 获取输入检测名额的超时秒数
    OUTPUT_ACQUIRE_TIMEOUT=10            # 获取输出检测名额的超时秒数

    SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE=true
    SAFE_SKIP_THRESHOLD=0.8

    FLASK_BACKLOG=128                    # TCP 连接 backlog 大小
"""

import os
import itertools
import threading
from flask import Flask, request, jsonify
from defend.prompt_detection import Detector
from defend.safety_detector import SafetyDetector
import torch
from gpu_utils import parse_gpu_ids


app = Flask(__name__)


# ───────────────────────── 固定配置 ─────────────────────────

# 敏感度固定为 medium，不再从请求体读取
FIXED_SENSITIVITY = "low"

# 输入检测 / 输出检测最大并发数
# 注意：这是单进程内的并发控制。
# 如果 gunicorn workers=2，则总并发上限约等于 2 * INPUT_MAX_CONCURRENCY。
_cuda_device_count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0
INPUT_GPU_IDS = parse_gpu_ids(os.environ.get("INPUT_DETECTOR_GPU_IDS"), _cuda_device_count)

INPUT_DETECTOR_REPLICAS_PER_GPU = max(1, int(os.environ.get("INPUT_DETECTOR_REPLICAS_PER_GPU", 1)))

_default_concurrency = (
    max(1, len(INPUT_GPU_IDS) * INPUT_DETECTOR_REPLICAS_PER_GPU)
    if INPUT_GPU_IDS else 8
)
INPUT_MAX_CONCURRENCY = int(os.environ.get("INPUT_MAX_CONCURRENCY", 0)) or _default_concurrency 
OUTPUT_MAX_CONCURRENCY = int(os.environ.get("OUTPUT_MAX_CONCURRENCY", 0)) or _default_concurrency 

# 获取推理名额的等待时间，超过则自动降级为安全放行
INPUT_ACQUIRE_TIMEOUT = float(os.environ.get("INPUT_ACQUIRE_TIMEOUT", 30))
OUTPUT_ACQUIRE_TIMEOUT = float(os.environ.get("OUTPUT_ACQUIRE_TIMEOUT", 30))

# TCP 连接 backlog（排队等待处理的连接数上限）
# 注意：Flask 内置开发服务器不支持 backlog 参数。
# 生产环境请使用 gunicorn（参见文件头部注释），通过 --backlog 设置。
FLASK_BACKLOG = int(os.environ.get("FLASK_BACKLOG", 128))

# 是否在输入检测高置信安全时跳过输出检测
SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE = (
    os.environ.get("SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE", "true").lower() == "true"
)

# 输入检测 p_safe 达到该阈值时，可跳过输出检测
SAFE_SKIP_THRESHOLD = 0.5


# ───────────────────────── 检测器进程级单例缓存 ─────────────────────────

# 输入检测器：按 GPU 维护一个小池，避免高并发集中到单个 cuda:0
_input_detectors = []
_input_detector_cycle = None
_input_detector_lock = threading.Lock()

# 输出检测器：全局只缓存一份
_output_detector = None
_output_detector_lock = threading.Lock()


# ───────────────────────── 推理并发闸门 ─────────────────────────

# 控制同一进程内同时进入输入检测推理的请求数
_input_infer_sem = threading.BoundedSemaphore(INPUT_MAX_CONCURRENCY)

# 控制同一进程内同时进入输出检测推理的请求数
_output_infer_sem = threading.BoundedSemaphore(OUTPUT_MAX_CONCURRENCY)


class ServiceBusyError(RuntimeError):
    """推理服务繁忙，无法在指定时间内获取并发名额。"""
    pass


def _get_bool(data, key, default=False):
    """从请求 JSON 中安全读取 bool。"""
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _validate_request_data(data):
    """校验请求体。返回 (ok, error_message)。"""
    if not data:
        return False, "请求体必须是 JSON"

    if "user_input" not in data:
        return False, "缺少必填字段: user_input"

    if "model_response" not in data:
        return False, "缺少必填字段: model_response"

    if not isinstance(data["user_input"], str):
        return False, "user_input 必须是字符串"

    if not isinstance(data["model_response"], str):
        return False, "model_response 必须是字符串"

    if not data["user_input"].strip():
        return False, "user_input 不能为空"

    if not data["model_response"].strip():
        return False, "model_response 不能为空"

    return True, None


def _build_input_detector(use_bert=False, llm_api_url=None, llm_api_model=None,
                          device=None):
    kwargs = {"use_bert": use_bert}

    if llm_api_url is not None:
        kwargs["llm_api_url"] = llm_api_url

    if llm_api_model is not None:
        kwargs["llm_api_model"] = llm_api_model

    if device is not None:
        kwargs["device"] = device

    detector = Detector(**kwargs)
    detector.set_bert()
    detector.set_sensitivity(FIXED_SENSITIVITY)
    return detector


def _get_input_detector(use_bert=False, llm_api_url=None, llm_api_model=None):
    """
    获取常驻输入 Detector。

    敏感度固定为 medium，因此当前进程只初始化一份 Detector。
    注意：use_bert、llm_api_url、llm_api_model 只在首次初始化时生效。
    后续请求会复用同一实例，不会因为请求体参数变化而重新构造。
    """
    global _input_detectors, _input_detector_cycle

    if _input_detectors:
        return next(_input_detector_cycle)

    with _input_detector_lock:
        if _input_detectors:
            return next(_input_detector_cycle)

        devices = [f"cuda:{gpu_id}" for gpu_id in INPUT_GPU_IDS] if INPUT_GPU_IDS else [None]
        detectors = []
        for device in devices:
            for _ in range(INPUT_DETECTOR_REPLICAS_PER_GPU):
                detectors.append(_build_input_detector(
                    use_bert=use_bert,
                    llm_api_url=llm_api_url,
                    llm_api_model=llm_api_model,
                    device=device,
                ))

        _input_detectors = detectors
        _input_detector_cycle = itertools.cycle(_input_detectors)
        device_names = [str(detector.device) for detector in _input_detectors]
        print(f"[预热] 输入检测器池就绪: sensitivity={FIXED_SENSITIVITY}, devices={device_names}")

        return next(_input_detector_cycle)


def _get_output_detector(constitution_file=None, api_key=None,
                         base_url=None, model_name=None):
    """
    获取常驻输出 SafetyDetector。

    当前进程只初始化一份 SafetyDetector。
    注意：constitution_file、api_key、base_url、model_name 只在首次初始化时生效。
    后续请求会复用同一实例。
    """
    global _output_detector

    if _output_detector is not None:
        return _output_detector

    with _output_detector_lock:
        if _output_detector is not None:
            return _output_detector

        kwargs = {}

        if constitution_file is not None:
            kwargs["constitution_file"] = constitution_file

        if api_key is not None:
            kwargs["api_key"] = api_key

        if base_url is not None:
            kwargs["base_url"] = base_url

        if model_name is not None:
            kwargs["model_name"] = model_name

        _output_detector = SafetyDetector(**kwargs)
        print("[预热] 输出检测器就绪")

        return _output_detector


def _run_input_detection(text, use_bert=False,
                         llm_api_url=None, llm_api_model=None):
    """
    执行输入检测。

    返回结构:
        {
            "is_safe": bool,
            "risk_level": str,
            "category": str,
            "confidence": float,
            "p_safe": float,
            "p_attack": float,
            "decision_by": str,
            "detection_error": str | None
        }
    """
    detector = _get_input_detector(
        use_bert=use_bert,
        llm_api_url=llm_api_url,
        llm_api_model=llm_api_model,
    )

    results = detector.detect_with_bert_group(text)

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

    return {
        "is_safe": is_safe,
        "risk_level": final_result.get(
            "mapped_risk_level",
            "无风险" if is_safe else "未知风险",
        ),
        "category": final_result.get(
            "mapped_category",
            "正常" if is_safe else "未知分类",
        ),
        "confidence": final_result.get("confidence_score", 0.0),
        "p_safe": final_result.get(
            "p_safe",
            probs_dict.get("安全", 1.0 if is_safe else 0.0),
        ),
        "p_attack": final_result.get(
            "p_attack",
            probs_dict.get("潜在风险提示词", 0.0 if is_safe else 1.0),
        ),
        "decision_by": final_result.get("final_decision_by", "unknown"),
        "detection_error": detection_error,
    }


def _run_input_detection_guarded(text, use_bert=False,
                                 llm_api_url=None, llm_api_model=None):
    """
    带并发控制的输入检测。

    如果当前输入检测推理已满载，则在 INPUT_ACQUIRE_TIMEOUT 秒后抛出 ServiceBusyError。
    """
    acquired = _input_infer_sem.acquire(timeout=INPUT_ACQUIRE_TIMEOUT)

    if not acquired:
        raise ServiceBusyError("输入检测服务繁忙，请稍后重试")

    try:
        return _run_input_detection(
            text=text,
            use_bert=use_bert,
            llm_api_url=llm_api_url,
            llm_api_model=llm_api_model,
        )
    finally:
        _input_infer_sem.release()


def _run_output_detection(text, answer,
                          constitution_file=None,
                          api_key=None,
                          base_url=None,
                          model_name=None):
    """
    执行输出检测。
    """
    detector = _get_output_detector(
        constitution_file=constitution_file,
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
    )

    return detector.detect(text=text, answer=answer)


def _run_output_detection_guarded(text, answer,
                                  constitution_file=None,
                                  api_key=None,
                                  base_url=None,
                                  model_name=None):
    """
    带并发控制的输出检测。

    如果当前输出检测推理已满载，则在 OUTPUT_ACQUIRE_TIMEOUT 秒后抛出 ServiceBusyError。
    """
    acquired = _output_infer_sem.acquire(timeout=OUTPUT_ACQUIRE_TIMEOUT)

    if not acquired:
        raise ServiceBusyError("输出检测服务繁忙，请稍后重试")

    try:
        return _run_output_detection(
            text=text,
            answer=answer,
            constitution_file=constitution_file,
            api_key=api_key,
            base_url=base_url,
            model_name=model_name,
        )
    finally:
        _output_infer_sem.release()


@app.route("/health", methods=["GET"])
def health():
    """
    健康检查接口。

    注意：这里只表示 Flask 进程存活，不代表检测器一定已经预热完成。
    """
    return jsonify({
        "status": "ok",
        "fixed_sensitivity": FIXED_SENSITIVITY,
        "input_max_concurrency": INPUT_MAX_CONCURRENCY,
        "output_max_concurrency": OUTPUT_MAX_CONCURRENCY,
        "input_gpu_ids": INPUT_GPU_IDS,
        "input_detector_replicas_per_gpu": INPUT_DETECTOR_REPLICAS_PER_GPU,
        "input_detector_devices": [str(detector.device) for detector in _input_detectors],
        "skip_output_when_high_confidence": SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE,
        "safe_skip_threshold": SAFE_SKIP_THRESHOLD,
    })


@app.route("/api/prompt_defend", methods=["POST"])
def api_defend():
    """
    防御流水线：输入检测 → 输出检测。

    当前实际逻辑:
        1. 输入检测判定不安全:
            直接拦截，stopped_at = input_detection

        2. 输入检测判定安全，且开启高置信跳过输出检测，且 p_safe >= SAFE_SKIP_THRESHOLD:
            直接放行，跳过输出检测

        3. 输入检测判定安全，但未达到跳过条件:
            执行输出检测

    请求体:
        {
            "user_input": "用户的原始查询",           // 必填
            "model_response": "模型的回复内容",      // 必填

            "use_bert": false,                       // 可选，是否启用 BERT，默认 false

            "input_llm_api_url": "...",             // 可选，仅首次初始化输入检测器时生效
            "input_llm_api_model": "...",           // 可选，仅首次初始化输入检测器时生效

            "constitution_file": "...",             // 可选，仅首次初始化输出检测器时生效
            "detector_api_key": "...",              // 可选，仅首次初始化输出检测器时生效
            "detector_api_url": "...",              // 可选，仅首次初始化输出检测器时生效
            "detector_model_name": "..."            // 可选，仅首次初始化输出检测器时生效
        }
    """
    data = request.get_json(silent=True)

    ok, error_message = _validate_request_data(data)
    if not ok:
        return jsonify({"error": error_message}), 400

    user_input = data["user_input"]
    model_response = data["model_response"]

    # ── 第一步：输入检测 ──
    try:
        input_detection = _run_input_detection_guarded(
            text=user_input,
            use_bert=_get_bool(data, "use_bert", False),
            llm_api_url=data.get("input_llm_api_url"),
            llm_api_model=data.get("input_llm_api_model"),
        )

    except ServiceBusyError as e:
        print(f"[503] input_detection busy (timeout={INPUT_ACQUIRE_TIMEOUT}s)")
        return jsonify({"error": "service_busy"}), 503

    except Exception as e:
        return jsonify({"error": f"输入检测失败: {str(e)}"}), 500

    # 输入检测链路内部异常：不放行
    if input_detection.get("detection_error"):
        return jsonify({
            "error": f"输入检测异常: {input_detection['detection_error']}",
        }), 500

    # 输入不安全：直接拦截
    if not input_detection["is_safe"]:
        return jsonify({"is_safe": False})

    # 输入安全且高置信：可跳过输出检测
    if (
        SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE
        and input_detection.get("p_safe", 0.0) >= SAFE_SKIP_THRESHOLD
    ):
        return jsonify({"is_safe": True})

    # ── 第二步：输出检测 ──
    try:
        output_result = _run_output_detection_guarded(
            text=user_input,
            answer=model_response,
            constitution_file=data.get("constitution_file"),
            api_key=data.get("detector_api_key"),
            base_url=data.get("detector_api_url"),
            model_name=data.get("detector_model_name"),
        )

    except ServiceBusyError as e:
        print(f"[503] output_detection busy (timeout={OUTPUT_ACQUIRE_TIMEOUT}s)")
        return jsonify({"error": "service_busy"}), 503

    except Exception as e:
        return jsonify({"error": f"输出检测失败: {str(e)}"}), 500

    return jsonify({"is_safe": output_result.is_safe})


if __name__ == "__main__":
    port = int(os.environ.get("FLASK_PORT", 8099))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"

    # 注意：Flask 内置 server 不支持 backlog 参数。
    # 生产环境高并发请使用 gunicorn（见文件头部注释）。
    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug,
        threaded=True,
    )
