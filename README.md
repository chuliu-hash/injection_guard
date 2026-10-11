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
- 输出检测判不安全 → `stopped_at="output_detection"`（`model_response` 原样返回，由前端替换展示）。
- 输出检测用 `resources/CritiqueRequests.json` 的 **12 条宪法规则**（风险类别 S1~S12，中英双语），
  **一次 LLM 调用**整体判定（内容中/英/混排均可），模型回 `safe` 或 `unsafe` + 换行 + 类别编码（如 `unsafe\nS5`）；
  编码即命中类别（`failed_rules`，如 `["S5"]`；模型没给类别时为空但判定仍是不安全）。
  调用失败或回复解析不出判定词时**默认放行**（记 warning）。
- RAG 增强由请求体 `use_rag` 控制（缺省取启动默认 `ENABLE_RAG=false`）；Embedding 不可用时**降级继续**。
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
├── auth/                # 认证原语：bcrypt 口令哈希 + JWT 签发/校验
├── persistence/         # 持久化：用户 / 会话 / 消息模型、仓储、引擎（按 user_id 隔离）
└── cli/                 # 四个命令行入口（含 deploy 一键部署）
data/                    # 运行时 SQLite 库（默认，勿入库）
resources/               # 随代码分发的数据（宪法规则等）
scripts/                 # 手工联调客户端
tests/                   # pytest 单测
```

## 会话持久化

会话 / 消息 / 检测记录**保存在服务端数据库**（不再依赖浏览器 localStorage）：默认 SQLite 文件
（`DATABASE_URL`，见 `.env.example`），生产可切 PostgreSQL。所有读写按**登录用户**做隔离——
换设备不丢、多端可见、按账号隔离。前端仅缓存，权威数据在服务端。

```
GET    /api/auth/me                               # 当前登录用户
POST   /api/auth/register                         # 注册（ALLOW_REGISTRATION=false 时 403）
POST   /api/auth/login                            # 登录 -> access token（refresh 走 httpOnly Cookie）
POST   /api/auth/refresh                          # 用 refresh Cookie 换新 access token（轮换）
POST   /api/auth/logout                           # 退出（all_devices=true 吊销该用户全部令牌）

GET    /api/sessions                              # 会话列表（含消息数）
POST   /api/sessions                              # 新建会话
DELETE /api/sessions                              # 清空当前用户全部会话
GET    /api/sessions/{id}                         # 会话详情（含消息与检测记录）
PATCH  /api/sessions/{id}                         # 重命名
DELETE /api/sessions/{id}                         # 删除（级联消息与检测记录）
POST   /api/sessions/{id}/messages                # 追加消息
PATCH  /api/sessions/{id}/messages/{mid}          # 更新消息（内容/状态/错误）
PUT    /api/sessions/{id}/messages/{mid}/detection# 写入 / 覆盖检测记录
```

## 登录鉴权

除 `/health` 与 `/api/auth/*` 外，所有接口都需 **`Authorization: Bearer <access token>`**，
缺失 / 过期 / 被吊销返回 `401`。令牌方案：

- **access token**（JWT / HS256，默认 30 分钟）：体返回，前端放内存，随请求头发送；
- **refresh token**（默认 7 天）：以 **httpOnly Cookie** 下发（JS 不可读，抗 XSS），
  仅用于 `/api/auth/refresh` 换取新 access token，且每次刷新**轮换**；
- 两者都带 `ver`（用户令牌版本），与 `users.token_version` 不符即失效——「退出所有设备」/改密可立即吊销。

口令用 **bcrypt** 加盐哈希存储（`auth/passwords.py`），明文永不落库。**生产必须设置
`JWT_SECRET_KEY`**（`openssl rand -hex 32`），否则沿用不安全的开发默认值并启动告警；
HTTPS 部署请置 `COOKIE_SECURE=true`。

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

# 输入检测 LLM（默认 localhost:8094；--service 决定读 configs/services.env 的哪一块推理参数）
python -m injection_guard.cli.serve_llm --model_path ./models/LLM/Qwen_prompt_injection --port 8094 --service input

# 输出检测 LLM（默认 :8095；权重路径以 configs/deploy.env 的 OUTPUT_MODEL_PATH 为准）
python -m injection_guard.cli.serve_llm --model_path ./models/LLM/Qwen3-8b --port 8095 --service output

# 普通聊天 LLM（默认 :8097，用于生成 model_response；权重路径见 CHAT_MODEL_PATH）
# 也可用云端：在 .env 配 CHAT_API_URL + CHAT_API_KEY + CHAT_MODEL，则无需本地部署
python -m injection_guard.cli.serve_llm --model_path ./models/LLM/<聊天模型> --port 8097 --service chat

# Embedding（默认 :8096，RAG 增强所需）
python -m injection_guard.cli.serve_embed --model_path Qwen3-Embedding-0.6B --port 8096
```

一键启动（读取 `configs/deploy.env`，按端口派生各 `*_API_URL` 并以子进程拉起各服务）：

```bash
injection-guard-deploy              # 或 python -m injection_guard.cli.deploy
injection-guard-deploy --dry-run    # 只打印派生环境变量与启动命令，不启动
injection-guard-deploy --only input,output
injection-guard-deploy --host 0.0.0.0 --url-host 10.0.0.5
```

也可以直接用 `scripts/` 下的启动脚本（每个服务一个，自动激活 conda 环境并读 `configs/deploy.env`，
环境变量可临时覆盖；详见 `scripts/README.md`）：

```bash
scripts/serve_all.sh                # = injection-guard-deploy
scripts/serve_input.sh              # 单独起输入检测 LLM（分终端调试用）
GPU_IDS=0,1 scripts/serve_embed.sh  # 临时指定 GPU
```

健康检查：各服务均暴露 `GET /health`；防御 API 的交互式文档在 `GET /docs`。

## 前端（Web AI Agent 聊天助手）

`frontend/` 是 Vue 3 + Vite + TypeScript 的聊天助手（**需登录**），接入后端安全护栏（流式）：

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173，/api 与 /health 代理到后端 :8099
npm run build        # 产出 frontend/dist，由防御 API 直接托管（访问 http://127.0.0.1:8099/）
npm run lint && npm run type-check && npm test
```

- 逐字流式输出；输入检测 / RAG 增强 / 输出检测结果**不内联显示**，统一在会话级「检测记录」抽屉查看；
- 输出检测判不安全时，收到 `output_detection` 事件即刻用拦截文案替换该条气泡（「先流式、结果出来再替换」）；
- 登录 / 注册（JWT：access token 存内存 + refresh httpOnly Cookie），路由守卫拦截未登录访问；
- RAG 开关、多会话历史（**服务端持久化**，见「会话持久化」；按用户隔离、多端同步）；
- 设置页「服务连通性」：一键测试输入 / 输出 / RAG / 聊天四个模型服务的连接与可用性。

详见 [`frontend/README.md`](frontend/README.md)。

## 请求 / 响应

```bash
# 非流式（一次性返回完整结果）
curl -X POST localhost:8099/api/prompt_defend -H 'Content-Type: application/json' \
  -d '{"user_input": "你好", "use_rag": false}'

# 流式（SSE）：逐事件推送 input_detection / rag / delta* / output_detection / done
curl -N -X POST localhost:8099/api/chat_stream -H 'Content-Type: application/json' \
  -d '{"user_input": "你好", "use_rag": false}'

# 连通性测试：对四个下游模型各发一次最小请求（body 可省略 = 全测）
curl -X POST localhost:8099/api/test_services
curl -X POST localhost:8099/api/test_services -H 'Content-Type: application/json' \
  -d '{"services": ["input", "chat"], "timeout": 5}'
```

- 请求：`user_input`（必填）；`model_response`（可选，缺省则调聊天接口生成）；`use_rag`（可选，缺省取启动默认）；
  `sensitivity`（可选，输入检测敏感度 `low`/`medium`/`high`，缺省取后端 `DEFAULT_SENSITIVITY`）；
  `temperature` / `max_tokens`（可选，聊天生成参数，由前端设置面板提供，缺省取后端 `Settings` 默认值；
  聊天服务端**不设生成长度上限**）；`safe_skip_threshold` / `skip_output_when_high_confidence`
  （可选，输出检测跳过策略，缺省取后端默认）。
- `/api/prompt_defend` 响应：`{is_safe, stopped_at, model_response, input_detection, rag, output_detection}`。
  `stopped_at` 取 `input_detection` / `output_detection` / `output_skipped`，走完全流程且输出检测通过时为 `null`。
  `model_response` **不因输出不安全而被替换**（那是"待检测的文本"，替换会破坏纯检测用法）。
- `/api/chat_stream` 响应为 `text/event-stream`，事件见下表；校验失败时返回普通 `400 {"error": ...}`。
- 错误一律 `{"error": ...}`：缺字段 → `400`；检测/生成异常 → `500`；并发繁忙 → `503 {"error":"service_busy"}`。
- `/api/test_services`（**连通性测试**，诊断用）：请求体可省略；可指定 `services`
  （`input`/`output`/`rag`/`chat` 子集，缺省全测）与 `timeout`（秒，0.5~60，默认 10）。
  **总是返回 200**，逐个服务成败在 `services` 里
  （`{ok, label, url, model, latency_ms, error}`）；仅参数非法（未知服务名 / 空列表 / timeout 越界）
  返回 `400 {"error": ...}`。它发真实最小请求（1 token 补全 / 单条 embedding），
  因此能同时验证「连得上」与「模型可用」；**不经过流水线并发闸门**。前端入口在设置页「服务连通性」。

| SSE event | data |
|-----------|------|
| `input_detection` | `{is_safe, risk_level, category, confidence, p_safe, p_attack, decision_by, sensitivity}` |
| `rag` | `{enabled, final_prompt, retrieved_contexts:[{prompt,response}]}` |
| `delta` | `{content}`（多次） |
| `output_detection` | `{is_safe, failed_rules, total_rules}`（12 条规则单次判定；`failed_rules` = 模型给出的命中类别编码） |
| `done` | `{is_safe, stopped_at, model_response}`（`stopped_at` 见上；输出不安全时为 `output_detection`） |
| `error` | `{error}`（`service_busy` 或异常文本） |

## 配置

均在 `src/injection_guard/config.py` 声明，分三份来源（各带 `.example` 模板）：

| 文件 | 读取方 | 内容 |
|------|--------|------|
| `.env` | `Settings`（全应用） | **访问项目 / 资源路径 / 连接模型** 必需项：`APP_*`、`LOG_LEVEL`、前端/CORS、`BERT_MODEL_DIR`/`RAG_DB_PATH`/`CRITIQUE_REQUESTS_PATH`、各 `*_API_URL`/`*_API_KEY`/`*_API_MODEL` |
| `configs/services.env` | `Settings` + `InferenceSettings`（`serve_llm`） | **只放影响检测的关键项**（输入截断 / RAG Top-K），按服务分块（**输入 / 输出 / Embedding / 聊天**）；敏感度默认值与检测生成长度（`16`）已内置为代码默认、无需写入；其余可调项（并发/超时/队列/缓存/GPU）见同名 `.example`（注释形式）。**不含客户端生成参数**（`sensitivity` / `temperature` / `max_tokens` 由前端设置面板随请求发送） |
| `configs/deploy.env` | `injection-guard-deploy` | 本地部署：模型路径、`HOST`/`URL_HOST`、端口、`WORKERS`/`MODE`/`REQUEST_TIMEOUT`/`GPU_IDS`/`DEPLOY_LOCAL_EMBED`/`DEPLOY_LOCAL_CHAT` |

优先级（高 → 低）：**环境变量 > 对应 `configs/*.env` > `.env` > 代码默认值**。
`configs/*.env` 用绝对路径读取（不受 CWD 影响）；`.env` 相对 CWD。
环境变量名大小写不敏感，沿用旧部署命名（如 `INPUT_MAX_CONCURRENCY`、`OUTPUT_DETECTOR_API_URL`、`INPUT_API_SERVER_MAX_NEW_TOKENS`）。
端口由 `injection-guard-deploy` 按 `configs/deploy.env` 统一派生并注入各 `*_API_URL`，无需在 `.env` 重复设置。

模型权重与数据路径通过 `BERT_MODEL_DIR` / `RAG_DB_PATH` **显式指定**（不再硬编码，也没有 `MODEL_ROOT` / `RESOURCES_DIR` 中间层）。
RAG 向量库缓存无需配置：与 `RAG_DB_PATH` 同目录、同主名（`.pt`），存在则加载、不存在则从 JSON 重构。

## 测试

```bash
pytest            # 纯函数 + FastAPI TestClient + 流水线编排（不依赖 GPU/模型）
```
