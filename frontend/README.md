# InjectionGuard 前端 · LLM 安全护栏控制台

Vue 3 + Vite + TypeScript 的对话控制台，接入后端**安全护栏**（`POST /api/chat_stream`，SSE 流式）：

- 逐字流式输出；
- **输入检测 / RAG 增强 / 输出检测**结果可视化，但**不在聊天区自动内联显示**，
  统一在会话级 **「检测记录」抽屉** 中查看；
- RAG 开关、多会话历史（**服务端持久化**：`/api/sessions`，按用户隔离、多端同步）。

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
首次访问会跳转登录页（`/login`）：可直接**注册**（后端 `ALLOW_REGISTRATION=false` 时需管理员建号）。

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
├── api/          # http 封装（含 Bearer 与 401 自动刷新）/ POST-SSE 解析 / 认证与会话接口
├── components/   # 会话列表、消息、输入框、检测记录抽屉等
├── layouts/      # 应用外壳
├── router/       # 路由（/login、/chat、/settings）+ 登录守卫
├── stores/       # Pinia：认证（内存 access token）、会话（服务端权威 + 内存缓存）、设置（localStorage）
├── types/        # 与后端 schemas / SSE 事件契约对应的类型
├── utils/        # Markdown 渲染（含 XSS 消毒）、格式化
└── views/        # LoginView / ChatView / SettingsView
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

## 会话持久化

会话 / 消息 / 检测记录保存在**服务端**（`/api/sessions`，见仓库根 README「会话持久化」），
浏览器不再持有权威数据：切换会话时按需拉取详情，流式增量只改本地缓存、结束时统一落库。

## 登录鉴权

- 登录 / 注册后，**access token 只存内存**（`api/authToken.ts`），随 `Authorization` 头发送；
  **refresh token 在 httpOnly Cookie**（JS 不可读），刷新页面时由 `stores/auth.ts` 自动恢复会话。
- `api/http.ts` 收到 401 会触发一次自动刷新并重试原请求；失败则清空登录态。
- 路由守卫（`router/index.ts`）拦截未登录访问并跳 `/login?redirect=...`；
  `App.vue` 对公开路由不套应用外壳，避免未登录时发起受保护请求。
