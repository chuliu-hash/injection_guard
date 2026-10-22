# scripts/ —— 各服务启动脚本

每个服务一个脚本，便于**分终端单独起**（排查 / 只起重训所需的那一个）。
端口、模型路径、workers 等统一取自 `configs/deploy.env`（唯一来源），无需在此重复配置。

## 脚本一览

| 脚本 | 服务 | 默认端口 | 说明 |
|------|------|----------|------|
| `serve_all.sh` | 全部（一键） | — | 等价于 `injection-guard-deploy`；按 `DEPLOY_LOCAL_EMBED` / `DEPLOY_LOCAL_CHAT` 决定是否拉起 embed / chat |
| `serve_defense.sh` | 防御 API | 8099 | 按端口派生并注入 `APP_PORT` 与各 `*_API_URL` |
| `serve_input.sh` | 输入检测 LLM | 8094 | `--service input`（读 `INPUT_API_SERVER_*`） |
| `serve_output.sh` | 输出检测 LLM | 8095 | `--service output`（读 `OUTPUT_API_SERVER_*`） |
| `serve_chat.sh` | 普通聊天 LLM | 8097 | `--service chat`（读 `CHAT_API_SERVER_*`） |
| `serve_embed.sh` | Embedding | 8096 | RAG 增强所需 |
| `_common.sh` | — | — | 公共部分（**勿直接执行**）：定位仓库根、激活 conda、载入 `deploy.env` |
| `run_p1.sh` | P1 验证 | — | 非交互式执行 ruff / pytest / 评测自检；失败汇总后退出 |

RAG 使用 Milvus 存储向量与文本，默认地址 `http://localhost:19530`。启动防御 API 前需先运行
Milvus Standalone；`serve_embed.sh` 只负责 Embedding 模型服务（8096），不包含 Milvus。

## 用法

```bash
# 一键起全部（推荐）
scripts/serve_all.sh
scripts/serve_all.sh --dry-run                 # 只打印派生环境变量与命令
scripts/serve_all.sh --only input,output       # 只起指定服务

# 单独起某一个
scripts/serve_input.sh
scripts/serve_embed.sh

# 任何服务脚本都可用环境变量临时覆盖（优先级：环境变量 > configs/deploy.env > 默认值）
GPU_IDS=0,1 scripts/serve_input.sh             # 指定 GPU
INPUT_PORT=9001 scripts/serve_input.sh         # 改端口
HOST=0.0.0.0 scripts/serve_defense.sh          # 对外监听

# 额外参数透传（后写覆盖前写）
scripts/serve_input.sh --public                # 起 cloudflared 隧道
scripts/serve_defense.sh --help

# P1 自动验证（无交互；默认使用 injection_guard conda 环境）
scripts/run_p1.sh
PYTHON_BIN=/path/to/python scripts/run_p1.sh
RUN_FRONTEND=1 scripts/run_p1.sh
```

脚本会：

1. `cd` 到仓库根（`.env` 相对 CWD 读取，必须从仓库根运行）；
2. 激活 conda 环境 `injection_guard`（用 `CONDA_ENV=<名>` 覆盖，`CONDA_ENV=` 置空则跳过激活）；
3. 若存在 `configs/deploy.env` 则载入其端口 / 模型路径等（已设置的环境变量优先）。

## 与 `injection-guard-deploy` 的关系

`serve_all.sh` 就是 `injection-guard-deploy` 的包装。分终端手动起多个服务时，用上面的
单服务脚本即可；`serve_defense.sh` 派生 `*_API_URL` 的逻辑与 `cli/deploy.py` 保持一致。

## 相关

- `test_api.py` —— 防御 API 的手工联调脚本（需先起服务）。
- 配置说明见仓库根 `README.md` 的「配置」章节与 `configs/*.env.example`。
