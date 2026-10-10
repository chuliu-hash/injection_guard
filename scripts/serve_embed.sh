#!/usr/bin/env bash
# 启动 Embedding 服务（OpenAI 兼容，:EMBED_PORT，默认 8096），RAG 增强所需。
# GPU 经 EMBED_GPU_IDS 生效（由 --gpu-ids 传入）；单请求超时取 --request-timeout。
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "$PYTHON_BIN" -m injection_guard.cli.serve_embed \
  --model_path "$EMBED_MODEL_PATH" \
  --host "$HOST" \
  --port "$EMBED_PORT" \
  --workers "$WORKERS" \
  --request-timeout "$REQUEST_TIMEOUT" \
  "${gpu_args[@]}" \
  "$@"
