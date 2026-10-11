# injection_guard

面向大模型（LLM）的**安全防御系统**：在聊天过程中自动拦截提示注入 / 越狱等风险。

- 检查**用户输入**是否包含注入、越狱等攻击意图；
- 检查**模型输出**是否违反安全规则；
- 自带网页版聊天助手（前端）与可替换的本地推理服务。

一句话：给聊天机器人套上「输入检测 → 生成回复 → 输出检测」的安全护栏，任一环节判定不安全即拦截。

## 主要功能

- 输入、输出**双向检测**，只放行通过检查的内容；
- 可选**检索增强（RAG）**，用安全示例引导模型；
- **网页聊天助手**：流式输出、多会话历史、每条回复的检测结果可查；
- **多用户账号**：注册登录后各看各的数据；
- 模型服务地址可配置，本地部署或调用云端均可。

## 技术架构

- **后端**：Python 3.10 + FastAPI，JWT 登录鉴权（访问令牌 + HttpOnly 刷新 Cookie），SQLAlchemy 会话持久化（默认 SQLite，可切 PostgreSQL）。
- **检测流水线**：由 **LangGraph** 状态图编排 `输入检测 → RAG 检索 → 生成 → 输出检测`，条件边负责「输入命中即拦截」与「高置信跳过输出检测」，流式场景由状态图逐节点推送事件（SSE）。
- **RAG 检索**：基于 **LangChain** —— 自定义 `Embeddings` 适配本地向量服务、FAISS 向量库、Retriever 检索安全示例，检索失败自动降级为原始输入。
- **模型调用**：聊天生成、输入/输出检测的 LLM 调用统一走 LangChain `ChatOpenAI` + `ChatPromptTemplate`（本地服务或云端接口均可，OpenAI 兼容）。
- **推理服务**：本地自研的 OpenAI 兼容 LLM / Embedding 推理服务（FastAPI + transformers）。
- **前端**：Vue 3 + Vite + TypeScript。

## 目录一览

| 位置 | 说明 |
|------|------|
| `src/injection_guard/` | 后端服务（Python / FastAPI） |
| `frontend/` | 网页聊天助手（Vue 3 + Vite） |
| `models/` | 本地模型权重（需自行放置） |
| `configs/` | 运行配置（端口、模型路径、服务参数） |
| `scripts/` | 各服务的启动脚本 |
| `tests/` | 自动化测试 |

## 检测模型

系统可用的开源输入检测模型和输出检测模型。**模型权重不随仓库分发**，请下载后放到对应目录：

| 用途 | 放到的目录 | HuggingFace 地址 |
|------|-----------|------------------|
| 输入检测 LLM | `models/LLM/Qwen_prompt_injection` | [xiangxinai/Xiangxin-Guardrails-Text](https://huggingface.co/xiangxinai/Xiangxin-Guardrails-Text) |
| 输出检测 LLM | `models/LLM/Qwen3-8b` | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) |
| 输入检测（BERT 分类器） | `models/BERT/mdeberta-prompt-injection-detector` | [chitsii/mdeberta-v2-base-prompt-injections](https://huggingface.co/chitsii/mdeberta-v2-base-prompt-injections)（旧名 `Fan1018/…`，会自动跳转） |
| 输入检测（BERT 分类器） | `models/BERT/protectai_deberta-v3-base-prompt-injection-v2` | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) |

下载示例（需先 `pip install huggingface_hub`；国内网络可加 `HF_ENDPOINT=https://hf-mirror.com` 加速）：

```bash
huggingface-cli download xiangxinai/Xiangxin-Guardrails-Text --local-dir models/LLM/Qwen_prompt_injection
huggingface-cli download Qwen/Qwen3-8B --local-dir models/LLM/Qwen3-8b
huggingface-cli download chitsii/mdeberta-v2-base-prompt-injections --local-dir models/BERT/mdeberta-prompt-injection-detector
huggingface-cli download protectai/deberta-v3-base-prompt-injection-v2 --local-dir models/BERT/protectai_deberta-v3-base-prompt-injection-v2
```

> `models/BERT/` 下另有 `fine_tuned_model`、`fine_tuned_model_base` 两个本地微调子模型，无公开下载地址。

## 环境准备

- Python 3.10+（建议用 conda 环境）
- Node.js 20+（只在开发 / 构建前端时需要）
- 需要本地跑大模型时：NVIDIA GPU + 已下载的模型权重

## 快速开始

在**仓库根目录**执行：

```bash
# 1) 安装依赖
pip install -e ".[dev]"

# 2) 首次配置
cp .env.example .env
cp configs/deploy.env.example configs/deploy.env
```

按需修改端口与模型路径。**生产环境请在 `.env` 中设置 `JWT_SECRET_KEY`**（可用 `openssl rand -hex 32` 生成）。

### 启动服务

```bash
injection-guard-deploy            # 一键启动全部服务
injection-guard-deploy --dry-run  # 只查看将要执行的命令，不真正启动
scripts/serve_all.sh              # 等价写法
```

启动完成后打开 **http://localhost:8099**，**注册一个账号**即可开始使用。

### 默认端口

| 服务 | 端口 |
|------|------|
| 防御 API + 网页前端 | 8099 |
| 输入检测 LLM | 8094 |
| 输出检测 LLM | 8095 |
| 向量检索（RAG） | 8096 |
| 聊天 LLM | 8097 |

### 分服务启动（便于单独调试）

```bash
scripts/serve_input.sh    # 输入检测 LLM
scripts/serve_output.sh   # 输出检测 LLM
scripts/serve_embed.sh    # 向量检索
scripts/serve_chat.sh     # 聊天 LLM
GPU_IDS=0,1 scripts/serve_embed.sh   # 临时指定 GPU
```

### 只用云端模型

不想在本地跑大模型时，在 `.env` 里配置聊天服务的地址与密钥即可，无需启动聊天服务（8097）。

### 前端开发（可选）

```bash
cd frontend
npm install
npm run dev        # http://127.0.0.1:5173（需后端已启动）
npm run build      # 构建产物 frontend/dist，由后端在 :8099 直接托管
```

## 使用说明

1. 打开 **http://localhost:8099**（开发模式下为 5173）；
2. 首次使用先**注册**账号并登录；
3. 输入消息开始聊天——被判定不安全的输入会被拦截，不安全的回复会被替换为拦截提示；
4. 点击「检测记录」查看每条回复对应的检查结果；
5. 「设置」页可测试各模型服务的连通性、调整检索增强与检测敏感度。

交互式接口文档：**http://localhost:8099/docs**。

## 测试

```bash
pytest                      # 后端测试
cd frontend && npm test     # 前端测试
```

## 常见问题

- **打开页面跳到登录页 / 提示 401**：正常现象，先注册或登录即可。
- **首次启动很慢**：首次需要加载模型权重，请耐心等待。
- **提示找不到模型**：把模型权重放到 `models/` 下，并在 `configs/deploy.env` 中确认路径。
- **只想先看看界面**：启动后端后直接访问 :8099 即可，前端无需单独启动。

## 配置说明

- `.env`：访问地址、连接信息、密钥等（复制 `.env.example`）。
- `configs/deploy.env`：本地部署的端口与模型路径（复制 `deploy.env.example`）。
- `configs/services.env`：各模型服务的检测参数（复制 `services.env.example`）。

所有示例文件均以 `.example` 结尾：复制为**同名文件**后按需修改即可，真实配置不会进入版本库。
