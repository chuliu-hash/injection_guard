# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI 安全防御系统：对 LLM 的**输入**与**输出**做多层提示注入 / 越狱检测，并提供 OpenAI 兼容的本地 LLM 与
Embedding 推理服务。全部代码在 `src/injection_guard/` 包内，统一为 **FastAPI**（无 Flask）。

- **防御 API**（`api/`，FastAPI :8099）—— 对外暴露 `POST /api/prompt_defend`（一次性返回）与
  `POST /api/chat_stream`（SSE 流式），执行端到端流水线：输入检测 →（可选 RAG 增强）→ 聊天生成 → 输出检测。
  同时可托管前端构建产物（SPA）。
- **推理服务**（`services/llm_server.py` :8094/:8095/:8097、`services/embed_server.py` :8096）—— OpenAI 兼容的本地
  LLM 与 Embedding 服务，被防御 API 与 RAG 模块调用。其中 :8097 是**普通聊天模型**，用于生成 `model_response`；
  `llm_server` 支持 `stream=true` 的 SSE 流式输出。
- **检测核心**（`defense/`）—— 纯逻辑：BERT 组 + LLM 复审（输入）、宪法规则并发检查（输出）、RAG。
- **聊天客户端**（`services/chat_client.py`）—— 调普通聊天接口生成回复（OpenAI SDK + base_url），含 `complete()` 与 `stream()`。
- **Web 前端**（`frontend/`）—— Vue 3 + Vite + TS + Pinia + Element Plus 聊天助手，接入护栏（流式）；构建产物由防御 API 托管。

## Running

```bash
# 防御 API（默认 0.0.0.0:8099）
python -m injection_guard.cli.serve_defense
# 或 console script / uvicorn factory：
injection-guard-defense
uvicorn injection_guard.api.app:create_app --factory --host 0.0.0.0 --port 8099

# 输入检测 LLM（默认 127.1.1.1:8094）
python -m injection_guard.cli.serve_llm --model_path ./model/LLMS/Qwen_prompt_injection --port 8094

# 输出检测 LLM（默认 :8095）
python -m injection_guard.cli.serve_llm --model_path ./model/LLMS/Qwen3-8B --port 8095

# 普通聊天 LLM（默认 :8097，请求未带 model_response 时用于生成）
python -m injection_guard.cli.serve_llm --model_path ./model/LLMS/Qwen3-8B --port 8097

# Embedding（默认 :8096，RAG 增强所需）
python -m injection_guard.cli.serve_embed --model_path Qwen3-Embedding-0.6B --port 8096

# 单测（纯函数 + FastAPI TestClient 冒烟，不依赖 GPU/模型）
pytest
pytest tests/test_defense_api.py::test_full_pipeline_returns_both_detections   # 单个用例

# 手工联调（需先起防御 API）
python scripts/test_api.py

# 前端（Vue 3 + Vite，需 node ≥20；npm 用 registry.npmmirror.com 镜像）
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173（/api、/health 代理到 :8099）
npm run build        # 产出 frontend/dist，由防御 API 在 / 托管
npm run lint && npm run type-check && npm test
```

一键启动（在部署目录运行）：`bash scripts/deploy_model.sh`。健康检查：各服务均暴露 `GET /health`；
防御 API 的交互式文档在 `GET /docs`。**上线前先构建前端**：`cd frontend && npm ci && npm run build`。

安装：`pip install -e ".[dev]"`（含 pytest/httpx/ruff），或 `pip install -r requirements.txt`（仅运行时）。

### 环境准备

- 需 conda 环境（`scripts/deploy_model.sh` 用 `LLM_Guards`）。**运行任何 Python 前先确认用哪个环境。**
- 模型权重**不入库**：用 `MODEL_ROOT` 等配置项指向外部路径（见下「配置」）。

## Architecture

### 部署拓扑与数据流

```
┌─ 防御 API (injection_guard.api, FastAPI :8099) ─────────────────────────┐
│  POST /api/prompt_defend（一次性） / POST /api/chat_stream（SSE 流式）    │
│    输入检测 ──►（可选 RAG 增强）──► 聊天生成 ──► 输出检测（可跳过）       │
│      │                │                  │              │                │
│      ▼                ▼                  ▼              ▼                │
│  BERT集成+LLM复审  Embedding 检索     普通聊天 LLM   宪法规则并发检查     │
└──────┼────────────────┼──────────────────┼──────────────┼────────────────┘
       ▼                ▼                  ▼              ▼
  输入检测 LLM(:8094) Embedding(:8096)  聊天 LLM(:8097) 输出检测 LLM(:8095)
       └────── injection_guard.services.llm_server / embed_server (FastAPI) ──────┘
```

`/api/chat_stream` 与 `/api/prompt_defend` 共用同一套流水线逻辑，只是前者把各阶段与 token 以 SSE 事件
（`input_detection` / `rag` / `delta`* / `output_detection` / `done` / `error`）逐条推出；`done` 的
`stopped_at` 语义与 `detect` 完全一致。防御 API 还可用 `SPAStaticFiles` 托管 `frontend/dist`（`/api`、`/health`、
`/docs` 因注册更早而优先，其余路径回退 `index.html` 以支持前端路由刷新）。

`services/llm_server.py` 是通用 OpenAI 兼容推理服务，输入/输出检测与聊天只是用不同 `--model_path`、
不同端口起了多份实例。

### 模块依赖关系

```
cli/serve_defense.py  →  api/app.py:create_app()
 ├── api/routes.py            (GET /health, POST /api/prompt_defend, POST /api/chat_stream[SSE])
 │    └── api/dependencies.py (DefenseService 单例 + 并发闸门 + Depends)
 │         ├── defense/prompt_detection.py (Detector → detect_with_bert_group)
 │         │    ├── defense/confidence_vote.py (confidence_weighted_vote)
 │         │    └── defense/risk_mapping.py    (RISK_LEVEL_MAPPING, CATEGORY_NAMES)
 │         ├── defense/safety_detector.py      (SafetyDetector.detect)
 │         ├── defense/rag.py                  (SCR_RAG_System → 可选 RAG 增强)
 │         └── services/chat_client.py         (ChatClient → complete() / stream())
 ├── api/schemas.py          (DefendRequest / DefendResponse / RagResult 等 Pydantic 模型)
 ├── url_utils.py            (join_endpoint：base URL + 端点路径)
 ├── SPAStaticFiles          (托管 frontend/dist，404 回退 index.html)
 └── config.py               (Settings + get_settings())

services/gpu.py (parse_gpu_ids / assign_process_gpu)    —— 被 llm_server / embed_server 共用

frontend/  (Vue 3 + Vite + TS + Pinia + Element Plus)   —— 独立 npm 工程，构建产物由防御 API 托管
```

关键点：`api/dependencies.py` **惰性导入** torch 与 Detector/SafetyDetector/RAG/ChatClient
（`TYPE_CHECKING` + 函数内 import），使 API 层可在无 GPU / 无 torch 的环境下被导入与测试。

### 防御流水线决策逻辑（`api/dependencies.py:DefenseService.detect`）

1. 校验请求体（必填 `user_input` 非空；`model_response` 可选，显式提供则须非空字符串）—— `validate_defend_request`。
2. **输入检测**：不安全 → `DefendOutcome(is_safe=False, stopped_at="input_detection")`（不生成、不做输出检测）。
3. **可选 RAG 增强**：`use_rag`（缺省取 `ENABLE_RAG`）为真时检索相似安全示例构建 `final_prompt`；
   失败（如 Embedding 未起）记告警并**降级为原始 `user_input`**，不阻断主流程。
4. **聊天生成**：请求未带 `model_response` 时调普通聊天接口（带并发闸门）生成回复；带了则直接用。
5. 输入安全且 `SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE=true` 且 `p_safe >= SAFE_SKIP_THRESHOLD` →
   跳过输出检测，`stopped_at="output_skipped"`（**仍返回 `model_response`**）。
6. 否则执行**输出检测**（"User Input" 用原始 `user_input`），`stopped_at=None`。
7. 获取并发名额超时 → 抛 `ServiceBusyError` → HTTP `503 {"error": "service_busy"}`；
   检测/生成链路内部异常 → `DetectionError` → HTTP `500 {"error": ...}`。

响应体为**完整详情**：`{is_safe, stopped_at, model_response, input_detection, rag, output_detection}`
（`stopped_at` 走完全流程时为 `null`；`rag` 未启用或降级时为 `null`）。

`DefenseService.stream_detect(...)` 是同一套分支的**同步生成器**版本：逐条 `yield (event, data)`，复用
`_run_input_detection_guarded` / `_run_rag` / `_run_chat_stream_guarded` / `_run_output_detection_guarded`；
请求带了 `model_response` 时不发 `delta`。`api/routes.py:/api/chat_stream` 用 `async def` + `StreamingResponse`，
在**工作线程**里跑该生成器、经 `asyncio.Queue` + `loop.call_soon_threadsafe` 桥回事件循环，并每 15s 发 `: ping` 心跳；
客户端断开时 `finally: gen.close()` 保证 `chat_sem` 释放。

## Key design details

### 进程级单例与并发控制（`api/dependencies.py`）

- 输入 `Detector` 按 GPU 维护一个小池（`INPUT_DETECTOR_GPU_IDS` × `INPUT_DETECTOR_REPLICAS_PER_GPU`），
  用 `itertools.cycle` 轮询取用，避免并发集中到单卡。
- 输出 `SafetyDetector`、聊天 `ChatClient`、RAG `SCR_RAG_System` 均**全局各缓存一份**（惰性初始化）。
- 请求体里的 `use_bert` / `llm_api_url` / 检测器 URL 等**只在首次初始化时生效**，之后复用同一实例。
- 三个 `BoundedSemaphore`（`INPUT_MAX_CONCURRENCY` / `OUTPUT_MAX_CONCURRENCY` / `CHAT_MAX_CONCURRENCY`）
  限制单进程内并发推理数，获取超时分别由 `INPUT_ACQUIRE_TIMEOUT` / `OUTPUT_ACQUIRE_TIMEOUT` /
  `CHAT_ACQUIRE_TIMEOUT` 控制。
- 路由用**同步 `def`**，交给 FastAPI 的 anyio 线程池执行 —— 这是为了让阻塞的 `BoundedSemaphore.acquire(timeout=...)`
  语义成立；**不要改成 `async def`**（否则阻塞事件循环）。**例外**：`/api/chat_stream` 用 `async def`，
  因为它自身不做阻塞工作（阻塞的流水线与闸门都在其内部的工作线程里跑），**不要**把该模式套用到其它路由。
- 并发上限默认 = GPU 数 × 每卡副本数。

### 输入检测（`defense/prompt_detection.py`）

- `Detector` 管理 4 个 BERT 模型的加载与推理；`set_bert()` 加载 `settings.bert_model_dir` 下的 4 个子目录
  （`fine_tuned_model`、`protectai_deberta-v3-base-prompt-injection-v2`、`fine_tuned_model_base`、
  `mdeberta-prompt-injection-detector`），缺失的会被跳过。
- `set_sensitivity(level)` 设定三档阈值（`low`/`medium`/`high`），同时设置 4 个参数：`dec_thresh`(BERT 判定阈值)、
  `llm_conf`(LLM 置信度阈值)、`llm_p`(触发 LLM 复审的模糊概率区间)、`llm_accept`(LLM 判定采纳阈值)。
  敏感度固定为 `FIXED_SENSITIVITY = "low"`。
- LLM 复审触发条件：BERT 置信度 < `llm_conf_threshold` OR 攻击概率落在 `llm_p_range` 内 OR `use_bert=False`。
- LLM 复审通过 `logprobs` 提取首 token 置信度，从回复中正则匹配 `S\d{1,2}` 风险代码；判为 unsafe 后还需
  置信度 ≥ `llm_accept_threshold` 才采纳，且**低风险自动降级为安全**。
- 每个 Detector 内部用 `self._infer_lock` 串行化 BERT 推理；HTTP 调用复用 `requests.Session` 连接池。

### 置信度加权投票（`defense/confidence_vote.py`）

`confidence_weighted_vote(prob_list, alpha=1.5, beta=0.4, gamma=0.5)`：
- 模型权重 = `|p - 0.5|^alpha`（越靠近 0/1 权重越大）。
- 跨模型方差 `U_inter` 对结果收缩（`beta`），趋向 0.5。
- 模型预测方向分歧（`ratio_pos` 落在 0.4~0.6）时施加惩罚（`gamma`）。
- 综合置信度 = `(1 - U_inter)^beta * avg_conf^alpha * output_certainty^gamma`。

### 输出检测（`defense/safety_detector.py`）

- 用 `openai` Python SDK（非原生 requests）调用 OpenAI 兼容 API。
- 加载 `settings.critique_requests_path`（默认 `resources/CritiqueRequests.json`，当前 5 条规则）。
- 每条规则独立调用 LLM，只要求输出 `safe`/`unsafe`；用**持久化** `ThreadPoolExecutor`（max_workers=max(规则数,4)）并发执行。
- 任一规则不通过即判定不安全；解析不到 `safe`/`unsafe` 或异常时**默认放行**（返回安全）。
- 参数优先级：**显式传入的参数 > settings**（与旧版「env 优先于参数」相反）。

### RAG 安全上下文（`defense/rag.py`）

- 首次运行从 `settings.rag_db_path` 构建向量库并缓存为 `settings.rag_embed_path`；后续直接加载缓存，
  支持 `add_new_attack_context()` 增量追加并持久化。
- 通过 OpenAI 兼容 Embedding API 取向量（`join_endpoint(EMBED_API_URL, "/embeddings")`），`util.cos_sim` 计算余弦相似度取 Top-K。
- `construct_safe_prompt()` 返回拼接了防御性 few-shot 示例的 `final_prompt`；`get_safe_rag_prompt()` 是便捷包装。
- **接入方式**：`DefenseService` 持有一个常驻 `SCR_RAG_System`，在流水线第 3 步按 `use_rag` / `ENABLE_RAG`
  决定是否调用；异常一律降级（不影响判定），`rag` 字段只回传 `{enabled, final_prompt, retrieved_contexts}`（剔除向量）。

### 聊天客户端（`services/chat_client.py`）

- `ChatClient` 用 `openai` SDK + `CHAT_API_URL`（base URL）调普通聊天接口，`complete()` 返回首条回复文本，
  `stream()` 逐个 `yield` 增量文本（`stream=True`）。
- 参数（`CHAT_MODEL` / `CHAT_TIMEOUT` / `CHAT_MAX_TOKENS` / `CHAT_TEMPERATURE`）全部走 `Settings`。
- 仅在请求未带 `model_response` 时被 `DefenseService._run_chat_guarded`（非流式）或 `_run_chat_stream_guarded`（流式）
  调用，均带 `CHAT_MAX_CONCURRENCY` 闸门；流式版本用 `try/finally` 保证生成器提前关闭时释放名额。

### 推理服务（`services/llm_server.py` / `services/embed_server.py`）

- `llm_server.py`：异步 FastAPI 端点 + `ThreadPoolExecutor(max_workers=1)` 串行访问 GPU，等待的请求自然排队；
  每请求有独立超时；支持 `logprobs`（`compute_transition_scores`）。支持 `full`/`lora` 两种加载模式与多 uvicorn worker
  （每 worker 懒加载独立模型副本）。模型路径/模式等由 `configure_model()` 在运行时经环境变量传入（多 worker 子进程需继承），
  因此这些**不**在 `Settings` 中声明；静态可调项（队列大小、token 上限等）走 `Settings`。
- `llm_server.py` 流式：`request.stream=true` 时 `create_chat_completion` 返回 `StreamingResponse`，用
  `transformers.TextIteratorStreamer` 在 `_model_executor` 内跑 `model.generate(streamer=...)`，逐 token 推
  OpenAI `chat.completion.chunk` SSE 帧，以 `data: [DONE]` 结束；`async` 包装经 `asyncio.Queue` + `call_soon_threadsafe`
  把 token 从工作线程桥回事件循环。**仍受 `_model_executor` 串行化约束**（语义不变）。
- `embed_server.py`：`create_app(model_path, request_timeout)` 工厂 + `run_server(...)`；`SentenceTransformer.encode` 放入线程池。
- 两者都用 `services/gpu.py:assign_process_gpu()` 做**跨进程** GPU 分配：通过 `tempfile` 下的 JSON + 自旋文件锁
  记录「进程 → GPU」映射，按负载均衡挑选 GPU，进程退出时释放。

## Configuration

所有可调项集中在 `src/injection_guard/config.py`（pydantic-settings `Settings`），可用环境变量或 `.env` 覆盖
（见 `.env.example`）。字段名小写下划线，**环境变量名大小写不敏感**，沿用旧部署命名
（如 `INPUT_MAX_CONCURRENCY`、`OUTPUT_DETECTOR_API_URL`、`API_SERVER_*`、`EMBED_GPU_IDS`）。

`get_settings()` 用 `@lru_cache` 返回单例；测试可用 `get_settings.cache_clear()` 覆盖。

模型权重与数据路径通过 `MODEL_ROOT` / `RESOURCES_DIR` 等配置项管理，**不再硬编码**（旧版是 `./user/Prompt_Injection/...`）。

| 用途 | 环境变量 | 默认值 |
|------|----------|--------|
| 防御 API 主机 / 端口 / 调试 / backlog | `APP_HOST` / `APP_PORT` / `APP_DEBUG` / `APP_BACKLOG` | `0.0.0.0` / `8099` / `false` / `128` |
| 日志级别 | `LOG_LEVEL` | `INFO` |
| 资源目录 / 模型根目录 | `RESOURCES_DIR` / `MODEL_ROOT` | 仓库 `resources/` / 仓库 `models/` |
| 输入检测并发上限 / 获取超时 | `INPUT_MAX_CONCURRENCY` / `INPUT_ACQUIRE_TIMEOUT` | GPU 数×副本数 / `30` |
| 输出检测并发上限 / 获取超时 | `OUTPUT_MAX_CONCURRENCY` / `OUTPUT_ACQUIRE_TIMEOUT` | 同上 / `30` |
| 输入检测 GPU / 每卡副本 | `INPUT_DETECTOR_GPU_IDS` / `INPUT_DETECTOR_REPLICAS_PER_GPU` | 全部可见 GPU / `1` |
| 敏感度 / 高置信跳过 / 跳过阈值 | `FIXED_SENSITIVITY` / `SKIP_OUTPUT_WHEN_HIGH_CONFIDENCE` / `SAFE_SKIP_THRESHOLD` | `low` / `true` / `0.5` |
| 输入检测 LLM | `INPUT_DETECTOR_API_URL` / `INPUT_DETECTOR_API_MODEL` | `127.1.1.1:8094/v1` / `Qwen3-input` |
| 输出检测 LLM | `OUTPUT_DETECTOR_API_URL` / `OUTPUT_DETECTOR_MODEL` / `OUTPUT_DETECTOR_TIMEOUT` / `OUTPUT_DETECTOR_MAX_TOKENS` | `127.1.1.1:8095/v1` / `Qwen_output` / `60` / `8` |
| 普通聊天 LLM | `CHAT_API_URL` / `CHAT_MODEL` / `CHAT_TIMEOUT` / `CHAT_MAX_TOKENS` / `CHAT_TEMPERATURE` / `CHAT_MAX_CONCURRENCY` / `CHAT_ACQUIRE_TIMEOUT` | `127.1.1.1:8097/v1` / `Qwen3-chat` / `120` / `1024` / `0.7` / GPU 数×副本数 / `120` |
| Embedding / RAG | `EMBED_API_URL` / `EMBED_MODEL_NAME` / `EMBED_GPU_IDS` | `127.1.1.1:8096/v1` / `embedding-model` / 全部可见 GPU |
| RAG 开关 / Top-K | `ENABLE_RAG` / `RAG_TOP_K` | `false` / `2` |
| 前端构建产物目录 / 跨域来源 | `FRONTEND_DIST_DIR` / `CORS_ORIGINS` | 仓库 `frontend/dist` / 空（不加 CORS） |
| 推理服务（`api_server`） | `API_SERVER_GPU_IDS`, `API_SERVER_SPLIT_ACROSS_GPUS`, `API_SERVER_MAX_QUEUE_SIZE`, `API_SERVER_MAX_INPUT_TOKENS`, `API_SERVER_MAX_NEW_TOKENS`, `API_SERVER_MEMORY_FRACTION`, `API_SERVER_EMPTY_CACHE_EVERY` | 见 `config.py` |

## Data files

| 文件 | 用途 |
|------|------|
| `resources/CritiqueRequests.json` | 输出检测的宪法规则（5 条），`{"critique_requests": [...]}` 格式 |
| `<MODEL_ROOT>/Prompt_Injection/overthink_prompt.json` | RAG 知识库源文件，每条含 `prompt` 和 `safe_response` |
| `<MODEL_ROOT>/Prompt_Injection/safety_context_compiled.pt` | RAG 运行时生成的向量库缓存（文本+向量），首次运行自动创建 |
| `<MODEL_ROOT>/Prompt_Injection/model/BERT/` | 4 个 BERT 分类模型目录 |
| `<MODEL_ROOT>/Prompt_Injection/model/LLMS/` | 本地 LLM 权重（输入/输出检测、Embedding） |

## 风险类别（`defense/risk_mapping.py`）

- 高风险：S2(敏感政治)、S3(国家形象)、S5(暴力犯罪)、S9(提示词攻击)
- 中风险：S4(伤害未成年人)、S6(违法犯罪)、S7(色情)、S8(歧视)、S10(辱骂)、S11(隐私)、S12(商业违规)
- 低风险：S1(一般政治话题) —— 以 `RISK_LEVEL_MAPPING` 为准

## Tests

`tests/` 下用 pytest：

- `test_confidence_vote.py` / `test_risk_mapping.py` / `test_url_utils.py`：纯函数断言。
- `test_defense_api.py`：`fastapi.testclient.TestClient` + 通过 `app.dependency_overrides[get_defense_service]`
  注入**桩 DefenseService**（不加载真实模型），覆盖健康检查、输入拦截、高置信跳过、走完输出检测、
  缺字段→400 `{"error":...}`、繁忙→503、检测异常→500。
- `test_pipeline.py`：`DefenseService.detect` 编排（子类化 + 桩替换各步骤），覆盖 RAG 开/关、RAG 降级、
  请求带 `model_response` 跳过聊天、高置信跳过、输出检测用原始 `user_input`。
- `test_stream_pipeline.py`：`DefenseService.stream_detect` 的事件序列（子类化 + 桩替换各步骤）。
- `test_stream_api.py`：`TestClient` 打 `/api/chat_stream`，断言 `text/event-stream`、事件序列、缺字段→400 JSON、繁忙→`error` 事件。

前端用 Vitest（`cd frontend && npm test`，jsdom 环境）：`api/sse.spec.ts`（分帧解析）、`stores/session.spec.ts`（会话 CRUD）、
`utils/markdown.spec.ts`（Markdown 渲染 + XSS 消毒）。

`pyproject.toml` 的 `[tool.pytest.ini_options]` 设了 `pythonpath=["src"]`；`tests/conftest.py` 也会把 `src/` 插入 `sys.path`。
