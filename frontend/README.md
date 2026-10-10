# injection_guard 前端 · Web AI Agent 聊天助手

Vue 3 + Vite + TypeScript 的聊天助手，接入后端**安全护栏**（`POST /api/chat_stream`，SSE 流式）：

- 逐字流式输出；
- **输入检测 / RAG 增强 / 输出检测**结果可视化，但**不在聊天区自动内联显示**，
  统一在会话级 **「检测记录」抽屉** 中查看；
- RAG 开关、多会话历史（浏览器 localStorage 持久化）。

## 环境要求

- Node.js ≥ 18（本机 v20）、npm ≥ 9。
- 依赖镜像已在 `.npmrc` 中固定为 `https://registry.npmmirror.com`。

## 开发

```bash
cd frontend
npm install
npm run dev          # http://127.0.0.1:5173，/api 与 /health 代理到后端 127.0.0.1:8099
```

后端需先启动：`python -m injection_guard.cli.serve_defense`（见仓库根 README）。

## 构建与部署

```bash
npm run build        # 类型检查 + 产出 frontend/dist
```

构建产物由**防御 API 直接托管**（`frontend_dist_dir`，默认 `frontend/dist`）：
启动后端后访问 `http://127.0.0.1:8099/` 即可。前后端同源，无需 CORS。

若前端与后端不同源部署，可设置 `VITE_API_BASE_URL`（构建期）并在后端配置 `CORS_ORIGINS`。

## 质量工具

```bash
npm run lint         # ESLint（flat config）
npm run type-check   # vue-tsc 严格类型检查
npm test             # Vitest（jsdom）
npm run format       # Prettier
```

## 目录结构

```
src/
├── api/          # http 封装 / POST-SSE 解析 / 业务接口
├── components/   # 会话列表、消息、输入框、检测记录抽屉等
├── layouts/      # 应用外壳
├── router/       # 路由（/chat、/settings）
├── stores/       # Pinia：会话历史、设置（持久化）
├── types/        # 与后端 schemas / SSE 事件契约对应的类型
├── utils/        # Markdown 渲染（含 XSS 消毒）、格式化
└── views/        # ChatView / SettingsView
```

## SSE 事件契约

`POST /api/chat_stream` 依次推送：

| event | data |
| --- | --- |
| `input_detection` | `{is_safe, risk_level, category, confidence, p_safe, p_attack, decision_by}` |
| `rag` | `{enabled, final_prompt, retrieved_contexts:[{prompt,response}]}` |
| `delta` | `{content}`（多次） |
| `output_detection` | `{is_safe, failed_rules, total_rules}`（12 条规则单次判定；`failed_rules` = 命中类别编码如 `["S5"]`） |
| `done` | `{is_safe, stopped_at, model_response}` |
| `error` | `{error}`（`service_busy` 或异常文本） |

> 因端点为 POST，浏览器原生 `EventSource` 不适用，故使用 `fetch` + `ReadableStream` 手动解析 SSE
> （见 `src/api/sse.ts`）。
