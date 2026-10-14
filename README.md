# InjectionGuard · LLM 安全护栏平台

面向大模型（LLM）的**安全护栏**：对输入与输出做双向检测，自动拦截提示注入 / 越狱等风险。

- 检查**用户输入**是否包含注入、越狱等攻击意图；
- 检查**模型输出**是否违反安全规则；
- 自带网页版**对话控制台**（前端）与可替换的本地推理服务。

一句话：给大模型应用套上「输入检测 → 生成回复 → 输出检测」的安全护栏，任一环节判定不安全即拦截。

## 主要功能

- 输入、输出**双向检测**，只放行通过检查的内容；
- 流式回复**每约 512 token 增量检测一次**，命中即中断生成（`OUTPUT_DETECT_INTERVAL_TOKENS` 可调 / 关闭）；
- **多轮记忆**：同一会话里回灌之前的问答（轮数与 token 预算可配），聊天不再是「单轮失忆」；
- 可选**检索增强（RAG）**，用安全示例引导模型；
- **对话控制台**：流式输出、多会话历史、每条回复的检测结果可查；
- **多用户账号**：注册登录后各看各的数据；
- 模型服务地址可配置，本地部署或调用云端均可。

## 技术架构

- **后端**：Python 3.10 + FastAPI，JWT 登录鉴权（访问令牌 + HttpOnly 刷新 Cookie），SQLAlchemy 会话持久化（默认 SQLite，可切 PostgreSQL）。
- **检测流水线**：由 **LangGraph** 状态图编排 `输入检测 → RAG 检索 → 生成 → 输出检测`，条件边负责「输入命中即拦截」与「高置信跳过输出检测」；检测节点挂了 `RetryPolicy`（只重试内部异常，并发繁忙的 503 不重试）；流式生成每约 512 token 增量检测一次、命中即中断，事件经状态图逐节点推送（SSE）。
- **RAG 检索**：基于 **LangChain** —— 自定义 `Embeddings` 适配本地向量服务、FAISS 向量库与官方索引持久化、按距离阈值过滤 + 去重检索安全示例；检索失败自动降级为原始输入，没命中则跳过上下文增强；检测记录里区分「未启用 / 已启用 / 未命中 / 降级（附失败原因）」。
- **模型调用**：聊天生成、输入/输出检测的 LLM 调用统一走 LangChain `ChatOpenAI` + `ChatPromptTemplate`（本地服务或云端接口均可，OpenAI 兼容）；输出检测用结构化 JSON 判定（解析失败先做一次修复重试），每次调用的耗时与 token 用量记入进程内指标（见 `GET /health` 的 `llm_metrics`）。
- **推理服务**：本地自研的 OpenAI 兼容 LLM / Embedding 推理服务（FastAPI + transformers）。
- **前端**：Vue 3 + Vite + TypeScript。

## 核心代码走读（想读源码的初学者看）

只想用起来可以跳过本节。下面四段是项目里最关键的代码，都摘自真实文件，可对照阅读。

**1）调用大模型：统一走 LangChain 的 `ChatOpenAI`**（`src/injection_guard/services/chat_client.py:84`）

```python
self.model = ChatOpenAI(
    model=self.model_name,        # 模型名
    base_url=self.base_url,       # 例如 http://localhost:8097/v1
    api_key=self.api_key,         # 本地服务不校验；留空时代码用 "EMPTY" 占位
    timeout=self.timeout,
    callbacks=[MetricsCallbackHandler("chat")],  # 记录耗时 / token 用量
)
# 发一次请求；bind() 只覆盖「本次调用」的参数（前端调温度 / 生成长度时用）
response = self._bind_model(max_tokens, temperature).invoke(messages)
```

- 为什么只给 `base_url`：OpenAI 兼容服务都遵循同一套 HTTP 协议（`/chat/completions`），
  所以换成本地服务或云端只改配置，代码不用动；
- 流式回复用 `model.stream(messages)` 逐块 `yield`，网页上的打字机效果就来自这里。

**2）多轮记忆：历史交给 `MessagesPlaceholder`，超预算自动裁剪**（`src/injection_guard/services/chat_client.py:100`）

```python
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [MessagesPlaceholder("history", optional=True), ("human", "{prompt}")]
)
# 先按 CHAT_HISTORY_MAX_TOKENS 裁到最近若干条再发给模型（本轮提问永远保留）
messages = trim_messages(messages, max_tokens=..., strategy="last", token_counter=approximate_tokens)
```

- 历史由**服务端**按 `session_id` 从数据库读取（只取已完成的消息、剔除本次这条提问）：
  前端只传 `session_id`、不传历史，客户端改不动上下文；
- 历史只参与「生成」，不参与输入检测：每次仍只检测**本轮输入**（信任边界见下）。

**3）RAG 检索：Embeddings → FAISS → 阈值过滤 + 去重**（`src/injection_guard/defense/rag.py:291`、`:353`）

```python
# 用已经算好的向量建索引（不重复调用 embedding 服务）
self._store = FAISS.from_embeddings(text_embeddings, self.embeddings, metadatas=metadatas)
# 检索时拿到 (Document, 距离)：按 RAG_MAX_DISTANCE 过滤，再按文本去重
scored = self._store.similarity_search_with_score(user_query, k=limit)
```

检索到的示例会被拼进最终 prompt（安全指令 + 相似示例 + 当前问题）；一条都没命中（或全部被
阈值过滤）时**不拼接空上下文**，直接用原始输入；检索失败同样降级，不阻断主流程。索引用 FAISS
官方 `save_local/load_local` 落盘到 `resources/jailbreak_prompts_index/`，旧版 `.pt` 缓存首次启动自动迁移。

**4）检测流水线：LangGraph 状态图**（`src/injection_guard/agent/pipeline.py:385`）

```python
builder = StateGraph(PipelineState)                  # 状态 = 请求参数 + 各阶段产物
builder.add_node("input_detection", bind(input_detection_node), retry_policy=retry_policy)
# 输入不安全 → 直接 END，不再执行后续阶段
builder.add_conditional_edges("input_detection", route_after_input, {"stop": END, "rag": "rag_retrieval"})
```

- 节点（Node）= 一个步骤，条件边（Conditional Edge）= 分支；流程一眼可见，每个节点可单独测试；
- `retry_policy` 只重试 `DetectionError`（下游抖动），`ServiceBusyError`（并发满 → 503）保持快速失败；
- 流式请求用 `stream_mode=["custom", "values"]` 逐节点把事件推给前端（SSE）。

按这个顺序读代码最省力：`services/chat_client.py`（调模型）→ `defense/rag.py`（检索）
→ `agent/pipeline.py`（编排）→ `api/dependencies.py`（并发控制与断流取消）。

## 目录一览

| 位置 | 说明 |
|------|------|
| `src/injection_guard/` | 后端服务（Python / FastAPI） |
| `frontend/` | 对话控制台（Vue 3 + Vite） |
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

> DeBERTa-V2 系分类器（如 `mdeberta-prompt-injection-detector`）依赖 `sentencepiece` 分词，已随
> `pip install -e .` 一并安装；若某目录只有 `spm.model` 而无 `tokenizer.json`，加载器会自动回退到慢速
> SentencePiece 分词器。单个模型加载失败只记 WARNING 并跳过，不影响其余分类器。

**BERT 分类器默认不参与检测**：`ENABLE_BERT=false`（默认）时输入检测只走 8094 的 LLM 复审；
在 `configs/services.env` 或环境变量里设为 `true`，服务才会加载 `models/BERT/` 下**全部**分类器目录做集成投票
（置信度不足或概率落在模糊区间时再触发 LLM 复审），加载发生在首次检测请求时。
选卡会跳过空闲显存低于 `INPUT_DETECTOR_MIN_FREE_MB`（默认 4096 MiB）的 GPU，避免与其他任务
抢显存导致 CUDA OOM；若候选卡都不足则本次不加载 BERT，输入检测退化为 LLM 复审。
模型在**服务启动时同步加载**（`WARMUP_ON_STARTUP=true`，默认）：启动日志会出现「输入检测器池加载完成」，
之后 `Application startup complete`；置为 `false` 则退回「首个检测请求时才加载」。

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
5. 「设置」页可测试各模型服务的连通性、调整检索增强与检测敏感度；
6. 同一会话里的追问会自动带上之前的对话（默认最近 10 轮 / 2048 token，`CHAT_HISTORY_MAX_TURNS=0` 可关闭）。

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
- `configs/services.env`：各模型服务的检测参数（复制 `services.env.example`），例如 `OUTPUT_DETECT_INTERVAL_TOKENS`。

常用的几个应用侧参数（写进 `.env` 或环境变量即可）：

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `CHAT_HISTORY_MAX_TURNS` | 10 | 回灌的历史轮数（0 = 关闭多轮记忆，退化为单轮） |
| `CHAT_HISTORY_MAX_TOKENS` | 2048 | 历史消息的 token 预算，超出按「保留最近消息」裁剪 |
| `RAG_MAX_DISTANCE` | 0 | 检索命中的最大距离（FAISS 欧氏距离，越小越相似；0 = 不过滤） |
| `DETECT_NODE_MAX_ATTEMPTS` | 2 | 检测节点最多尝试次数（1 = 不重试；只重试内部异常，503 不重试） |

**信任边界**：多轮记忆只把历史交给「生成」，输入检测永远只看本轮输入。历史消息在入库前
已经过当次输入检测或被标记为拦截，因此存入数据库的正文是「通过检测的内容」或拦截文案。

所有示例文件均以 `.example` 结尾：复制为**同名文件**后按需修改即可，真实配置不会进入版本库。
