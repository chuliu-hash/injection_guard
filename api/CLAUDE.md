# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI 安全防御系统，通过多层机制保护 LLM 免受提示注入、越狱等攻击。代码由两部分组成：

1. **防御 API 服务**（Flask，`app_pj.py`）—— 对外暴露 `/api/prompt_defend`，执行「输入检测 → 输出检测」流水线。
2. **模型推理服务**（FastAPI，`api_server.py` / `interact_embed.py`）—— OpenAI 兼容的本地 LLM 与 Embedding 服务，被防御 API 与 RAG 模块调用。

## Running

```bash
# 1) 防御 API（Flask，默认端口 8099）
python app_pj.py
# 或 flask --app app_pj run --host=0.0.0.0 --port=8099

# 2) 输入检测 LLM 服务（FastAPI + uvicorn，默认端口 8094）
python interact_pj.py --model_path ./model/LLMS/Qwen_prompt_injection --port 8094 --host 127.1.1.1

# 3) 输出检测 / 后端 LLM 服务（另一个端口，如 8095）
python interact_pj.py --model_path ./model/LLMS/Qwen3-8B --port 8095

# 4) Embedding 服务（供 RAG 使用，默认端口 8090，部署脚本用 8096）
python interact_embed.py --model_path Qwen3-Embedding-0.6B --port 8096

# 端到端测试（需先启动防御 API）
python test_api.py
```

一键启动（在 `model-shield-system` 根目录运行）：`bash deploy_model.sh`。

健康检查：防御 API 与两个推理服务均暴露 `GET /health`。

**没有** `requirements.txt` / `pyproject.toml`、没有测试框架、没有 lint 配置。运行需：flask, fastapi, uvicorn, torch, transformers, peft, sentence-transformers, openai, numpy, requests。

### ⚠️ 运行前必读

- **工作目录敏感，且 `api/` 布局下 `app_pj.py` 无法导入**：`prompt_detection.py`、`scr.py`、`safety_detector.py` 中的默认路径都是相对路径（`./user/Prompt_Injection/...`）；更关键的是 `defend/prompt_detection.py:13` 硬编码了 `from user.Prompt_Injection.defend.confidence_vote import ...` 这种以部署根为基准的导入。本仓库 `api/` 下没有 `user/` 包，所以 `from defend.prompt_detection import Detector` 会 `ModuleNotFoundError`，防御 API 在 `api/` 布局下无法启动。这些假设都源自原始部署目录 `model-shield-system/`（见 `deploy_model.sh`），需还原该目录结构（或改写这些导入）才能运行。
- 所有 `.py` 文件语法均可编译通过（已用 `python3 -m py_compile` 验证），文件头注释里提到的 `app_pj.py` 语法错误已不存在。

## Architecture

### 部署拓扑与数据流

```
┌─ 防御 API (app_pj.py, Flask :8099) ─────────────────────────────┐
│  POST /api/prompt_defend                                        │
│    输入检测 ──► 输出检测（可跳过）                              │
│      │                │                                         │
│      ▼                ▼                                         │
│  BERT 集成 + LLM复审   宪法规则并发检查                          │
└──────┼────────────────┼─────────────────────────────────────────┘
       │ HTTP           │ HTTP
       ▼                ▼
  输入检测 LLM 服务   输出检测 LLM 服务        Embedding 服务
  (interact_pj :8094) (interact_pj :8095)    (interact_embed :8096)
       └──────────── api_server.py (FastAPI, /v1/chat/completions) ─┘
```

`api_server.py` 是通用的 OpenAI 兼容推理服务，被 `interact_pj.py` 通过 `start_api_server()` 启动；输入/输出检测只是用不同 `--model_path`、不同端口起了两份实例。

### 模块依赖关系

```
app_pj.py  (Flask 入口)
 ├── defend.prompt_detection  (Detector → detect_with_bert_group)
 │    ├── defend.confidence_vote  (confidence_weighted_vote)
 │    └── defend.risk_mapping     (RISK_LEVEL_MAPPING, CATEGORY_NAMES)
 └── defend.safety_detector   (SafetyDetector.detect)
defend.scr  (SCR_RAG_System / get_safe_rag_prompt)  —— 独立模块，未被 app_pj 接入
gpu_utils.py  (parse_gpu_ids / assign_process_gpu)  —— 被 app_pj / api_server / interact_embed 共用
```

### 防御流水线决策逻辑（`app_pj.py:api_defend`）

1. 校验请求体（必填 `user_input`、`model_response`，均非空字符串）。
2. **输入检测**：不安全 → 直接返回 `{"is_safe": false}`（`stopped_at=input_detection`）。
3. 输入安全且 `SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE=true` 且 `p_safe >= SAFE_SKIP_THRESHOLD` → 跳过输出检测，直接 `{"is_safe": true}`。注意 `SAFE_SKIP_THRESHOLD` 是**硬编码常量 `0.5`**（`app_pj.py:93`），不可用环境变量配置；文件头 docstring 里写的 `SAFE_SKIP_THRESHOLD=0.8` 是过时注释。
4. 否则执行**输出检测**，返回 `{"is_safe": <output_result.is_safe>}`。
5. 检测器获取并发名额超时 → 返回 `503 {"error": "service_busy"}`（注意：文件头注释声称「繁忙时降级放行」，实际代码是返回 503，注释已过时）。

响应体只有 `is_safe`（或 `error`）；`test_api.py` 里读取的 `stopped_at` / `input_detection` / `output_detection` 字段与当前实现已不一致。

## Key design details

### 进程级单例与并发控制（`app_pj.py`）

与旧版「无状态」设计不同，当前实现**缓存检测器实例**：

- 输入 `Detector` 按 GPU 维护一个小池（`INPUT_GPU_IDS` × `INPUT_DETECTOR_REPLICAS_PER_GPU`），用 `itertools.cycle` 轮询取用，避免并发集中到单卡。
- 输出 `SafetyDetector` 全局缓存一份。
- 请求体里的 `use_bert` / `llm_api_url` / 检测器 URL 等**只在首次初始化时生效**，之后复用同一实例，不会因请求参数变化而重建。
- 用两个 `BoundedSemaphore`（`INPUT_MAX_CONCURRENCY` / `OUTPUT_MAX_CONCURRENCY`）限制单进程内并发推理数，获取超时由 `INPUT_ACQUIRE_TIMEOUT` / `OUTPUT_ACQUIRE_TIMEOUT` 控制。
- 并发上限默认 = GPU 数 × 每卡副本数；生产用 gunicorn+gevent（见文件头注释）时总上限约为 `workers × 单进程上限`。

### 输入检测（`defend/prompt_detection.py`）

- `Detector` 管理 4 个 BERT 模型的加载与推理；`set_bert()` 加载 `./user/Prompt_Injection/model/BERT/` 下的 4 个子目录（`fine_tuned_model`、`protectai_deberta-v3-base-prompt-injection-v2`、`fine_tuned_model_base`、`mdeberta-prompt-injection-detector`），缺失的会被跳过。
- `set_sensitivity(level)` 设定三档阈值（`low`/`medium`/`high`），同时设置 4 个参数：`dec_thresh`(BERT 判定阈值)、`llm_conf`(LLM 置信度阈值)、`llm_p`(触发 LLM 复审的模糊概率区间)、`llm_accept`(LLM 判定采纳阈值)。`app_pj.py` 中敏感度固定为 `FIXED_SENSITIVITY = "low"`（注意：代码注释写 medium，实际值是 low）。
- LLM 复审触发条件：BERT 置信度 < `llm_conf_threshold` OR 攻击概率落在 `llm_p_range` 内 OR `use_bert=False`。
- LLM 复审通过 `logprobs` 提取首 token 置信度，从回复中正则匹配 `S\d{1,2}` 风险代码；判为 unsafe 后还需置信度 ≥ `llm_accept_threshold` 才采纳，且**低风险自动降级为安全**。
- 每个 Detector 内部用 `self._infer_lock` 串行化 BERT 推理；HTTP 调用复用 `requests.Session` 连接池。

### 置信度加权投票（`defend/confidence_vote.py`）

`confidence_weighted_vote(prob_list, alpha=1.5, beta=0.4, gamma=0.5)`：
- 模型权重 = `|p - 0.5|^alpha`（越靠近 0/1 权重越大）。
- 跨模型方差 `U_inter` 对结果收缩（`beta`），趋向 0.5。
- 模型预测方向分歧（`ratio_pos` 落在 0.4~0.6）时施加惩罚（`gamma`）。
- 综合置信度 = `(1 - U_inter)^beta * avg_conf^alpha * output_certainty^gamma`。

### 输出检测（`defend/safety_detector.py`）

- 用 `openai` Python SDK（非原生 requests）调用 OpenAI 兼容 API。
- 加载 `CritiqueRequests.json` 的宪法规则（当前 5 条）。
- 每条规则独立调用 LLM，只要求输出 `safe`/`unsafe`；用**持久化** `ThreadPoolExecutor`（max_workers=max(规则数,4)）并发执行。
- 任一规则不通过即判定不安全；解析不到 `safe`/`unsafe` 或异常时**默认放行**（返回安全）。

### RAG 安全上下文（`defend/scr.py`）

- 首次运行从 `overthink_prompt.json` 构建向量库并缓存为 `safety_context_compiled.pt`；后续直接加载缓存，支持 `add_new_attack_context()` 增量追加并持久化。
- 通过 OpenAI 兼容 Embedding API 取向量（默认 `http://127.1.1.1:8096/v1/embeddings`），`util.cos_sim` 计算余弦相似度取 Top-K。
- `get_safe_rag_prompt()` 返回拼接了防御性 few-shot 示例的最终 prompt。注意：此模块目前未被 `app_pj.py` 接入，属独立能力。

### 推理服务（`api_server.py` / `interact_embed.py`）

- `api_server.py`：异步 FastAPI 端点 + `ThreadPoolExecutor(max_workers=1)` 串行访问 GPU，等待的请求自然排队；每请求有独立超时；支持 `logprobs`（`compute_transition_scores`）。支持 `full`/`lora` 两种加载模式与多 uvicorn worker（每 worker 懒加载独立模型副本）。
- `interact_embed.py`：同理的 Embedding 服务，`SentenceTransformer.encode` 放入线程池。
- 两者都用 `gpu_utils.assign_process_gpu()` 做**跨进程** GPU 分配：通过 `tempfile` 下的 JSON + 自旋文件锁记录「进程 → GPU」映射，按负载均衡挑选 GPU，进程退出时释放。

## Environment variables

| 用途 | 环境变量 | 默认值 |
|------|----------|--------|
| 防御 API 端口 / 调试 | `FLASK_PORT` / `FLASK_DEBUG` | `8099` / `false` |
| 输入检测并发上限 / 获取超时 | `INPUT_MAX_CONCURRENCY` / `INPUT_ACQUIRE_TIMEOUT` | GPU 数×副本数 / `30` |
| 输出检测并发上限 / 获取超时 | `OUTPUT_MAX_CONCURRENCY` / `OUTPUT_ACQUIRE_TIMEOUT` | 同上 / `30` |
| 输入检测 GPU / 每卡副本 | `INPUT_DETECTOR_GPU_IDS` / `INPUT_DETECTOR_REPLICAS_PER_GPU` | 全部可见 GPU / `1` |
| 高置信跳过输出检测 | `SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE` | `true` |
| 输入检测 LLM | `INPUT_DETECTOR_API_URL` / `INPUT_DETECTOR_API_MODEL` | `127.1.1.1:8094` / `Qwen3-input` |
| 输出检测 LLM | `OUTPUT_DETECTOR_API_URL` / `OUTPUT_DETECTOR_MODEL` / `OUTPUT_DETECTOR_TIMEOUT` / `OUTPUT_DETECTOR_MAX_TOKENS` | `127.1.1.1:8095` / `Qwen_output` / `60` / `8` |
| 推理服务（`api_server`） | `API_SERVER_GPU_IDS`, `API_SERVER_SPLIT_ACROSS_GPUS`, `API_SERVER_MAX_QUEUE_SIZE`, `API_SERVER_MAX_INPUT_TOKENS`, `API_SERVER_MAX_NEW_TOKENS`, `API_SERVER_MEMORY_FRACTION`, `API_SERVER_EMPTY_CACHE_EVERY` | 见源码 `_get_*_env` |
| Embedding 服务 GPU | `EMBED_GPU_IDS` | 全部可见 GPU |

## Data files

| 文件 | 用途 |
|------|------|
| `CritiqueRequests.json` | 输出检测的宪法规则（5 条），`{"critique_requests": [...]}` 格式 |
| `./user/Prompt_Injection/overthink_prompt.json` | RAG 知识库源文件，每条含 `prompt` 和 `safe_response` |
| `./user/Prompt_Injection/safety_context_compiled.pt` | RAG 运行时生成的向量库缓存（文本+向量），首次运行自动创建 |
| `./user/Prompt_Injection/model/BERT/` | 4 个 BERT 分类模型目录 |
| `./user/Prompt_Injection/model/LLMS/` | 本地 LLM 权重（输入/输出检测、Embedding） |

## 风险类别（`defend/risk_mapping.py`）

- 高风险：S2(敏感政治)、S3(国家形象)、S5(暴力犯罪)、S9(提示词攻击)
- 中风险：S4(伤害未成年人)、S6(违法犯罪)、S7(色情)、S8(歧视)、S10(辱骂)、S11(隐私)、S12(商业违规)
- 低风险：S1(一般政治话题) —— 注意 `RISK_LEVEL_MAPPING` 与旧文档描述不一致，以此文件为准
