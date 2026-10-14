# InjectionGuard · LLM 安全护栏平台

给大模型应用套上安全护栏：**输入检测 → 生成回复 → 输出检测**，任一环节判定不安全即拦截，
用于防御提示注入、越狱等风险。附带可替换的本地推理服务与网页版对话控制台。

## 特性

- **双向检测**：检查用户输入的攻击意图，也检查模型输出是否违反安全规则；流式回复每约 512 token
  增量检测一次，命中即中断生成。
- **开箱即用**：一条命令启动全部服务，自带 OpenAI 兼容的 LLM / Embedding 推理服务（含 Logprobs、
  `response_format`），也可只填云端模型地址。
- **多轮记忆**：同一会话自动带上之前的问答，轮数与 token 预算可配。
- **可选检索增强（RAG）**：用安全示例引导模型；没命中或检索失败自动降级，不阻断主流程。
- **多用户账号**：JWT 登录，会话与检测记录按用户隔离，存在服务端数据库里。
- **可观测**：每条回复都能查到检测结果，`GET /health` 暴露各模型调用的耗时与 token 用量。

## 技术栈

- **后端**：Python 3.10 + FastAPI；**LangGraph** 编排检测流水线，**LangChain** 做模型调用与 RAG 检索。
- **持久化**：SQLAlchemy（默认 SQLite，可切 PostgreSQL）。
- **推理服务**：transformers + FastAPI，提供 OpenAI 兼容接口。
- **前端**：Vue 3 + Vite + TypeScript。

## 目录结构

| 路径 | 说明 |
|------|------|
| `src/injection_guard/` | 后端服务（API / 检测流水线 / 推理服务） |
| `frontend/` | 对话控制台（Vue 3） |
| `models/` | 本地模型权重（需自行下载） |
| `configs/` | 运行配置（端口、模型路径、服务参数） |
| `scripts/` | 各服务的启动脚本，用法见 `scripts/README.md` |
| `tests/` | 后端自动化测试 |

## 快速开始

环境要求：Python 3.10+（建议 conda）；只在开发前端时需要 Node.js 20+；本地跑模型需要 NVIDIA GPU。

```bash
# 1) 安装依赖
pip install -e ".[dev]"

# 2) 准备配置
cp .env.example .env
cp configs/deploy.env.example configs/deploy.env

# 3) 启动全部服务
injection-guard-deploy            # 等价于 scripts/serve_all.sh
injection-guard-deploy --dry-run  # 只打印将要执行的命令，不真正启动
```

按需修改端口与模型路径；**生产环境务必在 `.env` 中设置 `JWT_SECRET_KEY`**（`openssl rand -hex 32`）。

启动后打开 `http://localhost:8099`，注册账号即可使用；交互式接口文档见 `http://localhost:8099/docs`。

### 默认端口

| 服务 | 端口 |
|------|------|
| 防御 API + 网页前端 | 8099 |
| 输入检测 LLM | 8094 |
| 输出检测 LLM | 8095 |
| 向量检索（RAG） | 8096 |
| 聊天 LLM | 8097 |

### 分开启动 / 只用云端模型

```bash
scripts/serve_input.sh   # 输入检测 LLM
scripts/serve_output.sh  # 输出检测 LLM
scripts/serve_embed.sh   # 向量检索
scripts/serve_chat.sh    # 聊天 LLM
GPU_IDS=0,1 scripts/serve_embed.sh   # 临时指定 GPU
```

不想在本地跑大模型时，在 `.env` 里填好聊天服务的地址与密钥即可，无需启动 8097。

### 前端开发（可选）

```bash
cd frontend
npm install
npm run dev     # http://127.0.0.1:5173，需要后端已启动
npm run build   # 产物 frontend/dist，由后端在 8099 直接托管
```

## 模型

**模型权重不随仓库分发**，下载后放到下表中的目录即可：

| 用途 | 目录 | 来源 |
|------|------|------|
| 输入检测 LLM | `models/LLM/Qwen_prompt_injection` | [xiangxinai/Xiangxin-Guardrails-Text](https://huggingface.co/xiangxinai/Xiangxin-Guardrails-Text) |
| 输出检测 LLM | `models/LLM/Qwen3-8b` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| 输入检测（BERT 分类器） | `models/BERT/mdeberta-prompt-injection-detector` | [chitsii/mdeberta-v2-base-prompt-injections](https://huggingface.co/chitsii/mdeberta-v2-base-prompt-injections) |
| 输入检测（BERT 分类器） | `models/BERT/protectai_deberta-v3-base-prompt-injection-v2` | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) |

```bash
huggingface-cli download xiangxinai/Xiangxin-Guardrails-Text --local-dir models/LLM/Qwen_prompt_injection
huggingface-cli download Qwen/Qwen3-8B --local-dir models/LLM/Qwen3-8b
huggingface-cli download chitsii/mdeberta-v2-base-prompt-injections --local-dir models/BERT/mdeberta-prompt-injection-detector
huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 --local-dir models/BERT/protectai_deberta-v3-base-prompt-injection-v2
```

国内网络可加 `HF_ENDPOINT=https://hf-mirror.com` 加速。`models/BERT/` 下另有本项目本地微调的
子模型（`fine_tuned_model`、`fine_tuned_model_base`），无公开下载地址。

**BERT 分类器默认不参与检测**（`ENABLE_BERT=false`）：此时输入检测只走 8094 的 LLM 复审；
设为 `true` 才会加载 `models/BERT/` 下的分类器做集成投票。加载发生在服务启动时（`WARMUP_ON_STARTUP`），
单个模型加载失败只记 WARNING 并跳过。

## 配置

示例文件复制为**同名文件**后按需修改，真实配置不会进入版本库：

- `.env`：访问地址、连接信息、密钥。
- `configs/deploy.env`：本地部署的端口与模型路径。
- `configs/services.env`：各模型服务的检测参数。

常用的几个应用侧参数（写到 `.env` 或环境变量即可）：

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `CHAT_HISTORY_MAX_TURNS` | 10 | 回灌的历史轮数（0 = 关闭多轮记忆） |
| `CHAT_HISTORY_MAX_TOKENS` | 2048 | 历史消息的 token 预算，超出按「保留最近消息」裁剪 |
| `RAG_MAX_DISTANCE` | 0 | 检索命中的最大距离（FAISS 欧氏距离，越小越相似；0 = 不过滤） |
| `DETECT_NODE_MAX_ATTEMPTS` | 2 | 检测节点最多尝试次数（1 = 不重试） |
| `OUTPUT_DETECT_INTERVAL_TOKENS` | 512 | 流式回复的增量检测间隔（0 = 关闭） |

## 测试

```bash
pytest                                  # 后端
cd frontend && npm test                 # 前端
ruff check src tests                    # lint（ruff format 格式化）
```

## 常见问题

- **打开页面就跳到登录页 / 提示 401**：正常现象，先注册或登录。
- **首次启动很慢**：首次要加载模型权重，请耐心等待。
- **提示找不到模型**：确认权重已放到 `models/` 下，并检查 `configs/deploy.env` 中的路径。
- **只想先看看界面**：启动后端后直接访问 :8099，前端无需单独启动。
