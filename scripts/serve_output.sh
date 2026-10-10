#!/usr/bin/env bash
# 启动输出检测 LLM（OpenAI 兼容，:OUTPUT_PORT，默认 8095）。
# 推理参数读 configs/services.env 的 OUTPUT_API_SERVER_* 块（--service output）。
# 额外参数透传，如：scripts/serve_output.sh --public
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "$PYTHON_BIN" -m injection_guard.cli.serve_llm \
  --model_path "$OUTPUT_MODEL_PATH" \
  --host "$HOST" \
  --port "$OUTPUT_PORT" \
  --workers "$WORKERS" \
  --mode "$MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --service output \
  "${gpu_args[@]}" \
  "$@"
