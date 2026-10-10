#!/usr/bin/env bash
# 启动普通聊天 LLM（OpenAI 兼容，:CHAT_PORT，默认 8097），用于生成 model_response。
# 推理参数读 configs/services.env 的 CHAT_API_SERVER_* 块（--service chat）。
#
# 注：默认用云端聊天（configs/deploy.env 的 DEPLOY_LOCAL_CHAT=false，走 .env 的
# CHAT_API_URL / CHAT_API_KEY），需要本地部署时再手动跑本脚本并设 DEPLOY_LOCAL_CHAT=true
# （或直接用 `injection-guard-deploy`，它会按该开关自动拉起）。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "$PYTHON_BIN" -m injection_guard.cli.serve_llm \
  --model_path "$CHAT_MODEL_PATH" \
  --host "$HOST" \
  --port "$CHAT_PORT" \
  --workers "$WORKERS" \
  --mode "$MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --service chat \
  "${gpu_args[@]}" \
  "$@"
