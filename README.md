# injection_guard

AI 安全防御系统：对 LLM 的**输入**与**输出**做多层提示注入 / 越狱检测，并提供 OpenAI 兼容的本地
LLM 与 Embedding 推理服务。

## 架构

`POST /api/prompt_defend` 是一条**端到端受控聊天流水线**：

```
user_input ─► 输入检测 ─►（可选 RAG 增强 → final_prompt）─► 聊天接口 ─► model_response ─► 输出检测
                  │                 │                          │
                  ▼                 ▼                          ▼
            BERT 集成 + LLM 复审   Embedding 检索(:8096)     普通聊天 LLM(:8097)
                  │                                            │
                  ▼                                            ▼
             输入检测 LLM(:8094)                          输出检测 LLM(:8095)
```

- 请求带 `model_response` 时**跳过聊天调用**（兼容旧「纯检测」用法）；未带则由聊天接口生成。
- 输入不安全 → 直接拦截（`stopped_at="input_detection"`）。
- 输入安全且高置信 → 可跳过输出检测（`stopped_at="output_skipped"`，仍返回 `model_response`）。
- RAG 增强由 `ENABLE_RAG` / 请求体 `use_rag` 控制（默认关）；Embedding 不可用时**降级继续**。
- 所有下游服务配置统一为 **base URL**（`http://host:port/v1`），端点路径由 `url_utils.join_endpoint` 拼接。

## 目录结构

```
src/injection_guard/
├── config.py            # pydantic-settings 集中配置
├── logging_config.py    # 统一日志
├── url_utils.py         # base URL + 端点路径拼接
├── api/                 # 防御 API（app 工厂 / 路由 / schemas / 服务层依赖）
├── defense/             # 检测核心（BERT 组 + LLM 复审 / 输出检测 / RAG）
├── services/            # 推理服务（LLM / Embedding）与跨进程 GPU 分配、聊天客户端
└── cli/                 # 三个命令行入口
resources/               # 随代码分发的数据（宪法规则等）
scripts/                 # 部署脚本与手工联调客户端
tests/                   # pytest 单测
```

## 安装

```bash
pip install -e ".[dev]"     # 含 pytest/httpx/ruff
# 或仅运行时：pip install -r requirements.txt
```

## 运行

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

# 普通聊天 LLM（默认 :8097，用于生成 model_response）
python -m injection_guard.cli.serve_llm --model_path ./model/LLMS/Qwen3-8B --port 8097

# Embedding（默认 :8096，RAG 增强所需）
python -m injection_guard.cli.serve_embed --model_path Qwen3-Embedding-0.6B --port 8096
```

一键启动：`bash scripts/deploy_model.sh`。健康检查：三个服务均暴露 `GET /health`；
防御 API 的交互式文档在 `GET /docs`。

## 前端（Web AI Agent 聊天助手）

`frontend/` 是 Vue 3 + Vite + TypeScript 的聊天助手，接入后端安全护栏（流式）：

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173，/api 与 /health 代理到后端 :8099
npm run build        # 产出 frontend/dist，由防御 API 直接托管（访问 http://127.0.0.1:8099/）
npm run lint && npm run type-check && npm test
```

- 逐字流式输出；输入检测 / RAG 增强 / 输出检测结果**不内联显示**，统一在会话级「检测记录」抽屉查看；
- RAG 开关、多会话历史（浏览器 localStorage 持久化）。

详见 [`frontend/README.md`](frontend/README.md)。

## 请求 / 响应

```bash
# 非流式（一次性返回完整结果）
curl -X POST localhost:8099/api/prompt_defend -H 'Content-Type: application/json' \
  -d '{"user_input": "你好", "use_rag": false}'

# 流式（SSE）：逐事件推送 input_detection / rag / delta* / output_detection / done
curl -N -X POST localhost:8099/api/chat_stream -H 'Content-Type: application/json' \
  -d '{"user_input": "你好", "use_rag": false}'
```

- 请求：`user_input`（必填）；`model_response`（可选，缺省则调聊天接口生成）；`use_rag`（可选）。
- `/api/prompt_defend` 响应：`{is_safe, stopped_at, model_response, input_detection, rag, output_detection}`。
- `/api/chat_stream` 响应为 `text/event-stream`，事件见下表；校验失败时返回普通 `400 {"error": ...}`。
- 错误一律 `{"error": ...}`：缺字段 → `400`；检测/生成异常 → `500`；并发繁忙 → `503 {"error":"service_busy"}`。

| SSE event | data |
|-----------|------|
| `input_detection` | `{is_safe, risk_level, category, confidence, p_safe, p_attack, decision_by}` |
| `rag` | `{enabled, final_prompt, retrieved_contexts:[{prompt,response}]}` |
| `delta` | `{content}`（多次） |
| `output_detection` | `{is_safe, failed_rules, total_rules}` |
| `done` | `{is_safe, stopped_at, model_response}` |
| `error` | `{error}`（`service_busy` 或异常文本） |

## 配置

所有可调项集中在 `src/injection_guard/config.py`，可用环境变量或 `.env` 覆盖
（见 `.env.example`）。环境变量名大小写不敏感，沿用旧部署命名（如
`INPUT_MAX_CONCURRENCY`、`OUTPUT_DETECTOR_API_URL`、`API_SERVER_*`）。

模型权重与数据路径通过 `MODEL_ROOT` / `RESOURCES_DIR` 等配置项管理，**不再硬编码**。

## 测试

```bash
pytest            # 纯函数 + FastAPI TestClient + 流水线编排（不依赖 GPU/模型）
```
