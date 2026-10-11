# Repository Guidelines

AI 安全防御系统：对 LLM 的输入/输出做提示注入、越狱检测，并提供 OpenAI 兼容的本地 LLM 与 Embedding 推理服务。后端 Python + FastAPI，前端 Vue 3 + Vite + TypeScript。

## Project Structure & Module Organization
- `src/injection_guard/` — 后端包：`config.py`（配置）、`api/`（FastAPI 工厂/路由/schemas）、`auth/`（bcrypt 口令 + JWT）、`defense/`（BERT + LLM 检测核心）、`services/`（推理服务/GPU/聊天客户端）、`persistence/`（用户/会话 ORM 与仓储）、`cli/`（四个命令行入口）。
- `tests/` — pytest 单测（`test_*.py`）。
- `frontend/` — Vue 3 应用：`frontend/src/{api,components,stores,views,utils,types}`，Vitest 用例为同目录 `*.spec.ts`。
- `resources/` — 随包数据（宪法规则等）；`configs/` — `.env` 分层配置；`scripts/` — 各服务启动脚本；`models/` — 本地权重（勿入库）。

## Data Persistence
- 会话 / 消息 / 检测记录存**服务端数据库**（`persistence/`，SQLAlchemy）；默认 SQLite（`DATABASE_URL`），生产可切 PostgreSQL。禁止新增浏览器 localStorage 存业务数据。
- 所有查询经 `SessionRepository` 按**登录用户**（`Depends(get_current_user_id)`）隔离；新增接口必须沿用该依赖，勿绕过用户过滤。
- 前端 `stores/session.ts` 仅做缓存与乐观更新：流式增量只改本地，结束时用 `patchMessage`/`putDetection` 落库。

## Authentication & Security
- 除 `/health` 与 `/api/auth/*` 外，所有接口须带 `Authorization: Bearer <access token>`；新增受保护路由用 `dependencies=[Depends(require_user)]`（`api/security.py`）。
- 口令只存 bcrypt 哈希（`auth/passwords.py`），明文永不落库/进日志；access token 由前端内存持有，refresh token 走 httpOnly Cookie。
- 令牌带 `ver`（`users.token_version`）：递增即吊销该用户全部令牌（`/api/auth/logout {all_devices:true}` 已实现）。
- **生产必须设 `JWT_SECRET_KEY`**（`openssl rand -hex 32`），HTTPS 环境置 `COOKIE_SECURE=true`；勿把密钥写入代码或提交入库。

## Build, Test, and Development Commands
- `pip install -e ".[dev]"` — 安装（含 pytest/httpx/ruff）。
- `pytest` — 运行全部单测；`pytest tests/test_defense_api.py::test_full_pipeline_returns_both_detections` 跑单个用例。
- `ruff check src tests`（`ruff format` 格式化）。
- `injection-guard-deploy` / `scripts/serve_*.sh` — 一键或分服务启动（见 `scripts/README.md`）。
- `cd frontend && npm install`，`npm run dev`（:5173），`npm run build`（产出 `frontend/dist`），`npm run lint && npm run type-check && npm test`。

## Coding Style & Naming Conventions
- Python 3.10+，4 空格缩进，行宽 120（`pyproject.toml`）；模块/函数/变量 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE`。
- ruff lint：`E/F/I/W/B`（忽略 `E501`、`B008`，后者是 FastAPI `Depends` 惯用法）。
- 前端：Prettier（无分号、单引号、宽度 100、尾逗号）+ ESLint，组件文件用 `PascalCase.vue`。

## Testing Guidelines
- 框架 pytest（`pythonpath=["src"]`，`-q`）；前端 Vitest + jsdom。测试不得依赖 GPU/真实模型——用桩替换 `DefenseService` 或 `dependency_overrides`。
- 命名 `test_<模块>.py` / `<name>.spec.ts`；新增行为请覆盖对应模块，无独立覆盖率门槛。

## Commit & Pull Request Guidelines
- 提交信息用中文、祈使式、一句话概括（历史如“重构为 injection_guard 企业级包…”）；改动大时正文分点说明。
- PR 需写明目的与影响范围、关联 issue，附测试结果；涉及 UI 改动附截图。合并前确保 `pytest`、`ruff`、前端 `lint/type-check/test` 全通过。

## Security & Configuration Tips
- 密钥只放 `.env` / `configs/*.env`（已 gitignore），勿提交；示例见 `*.env.example`。
- 三层层级配置：环境变量 > `configs/deploy.env` > 默认值。
- 下游服务统一用 base URL（`http://host:port/v1`），端点由 `url_utils.join_endpoint` 拼接；勿随意升级 `torch`/`transformers` 等相互钳制的推理依赖（见 `pyproject.toml` 注释）。
