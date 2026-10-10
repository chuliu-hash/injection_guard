#!/usr/bin/env bash
# 启动输入检测 LLM（OpenAI 兼容，:INPUT_PORT，默认 8094）。
# 推理参数读 configs/services.env 的 INPUT_API_SERVER_* 块（--service input）。
# 额外参数透传（后写的覆盖前写的），如：scripts/serve_input.sh --public
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "$PYTHON_BIN" -m injection_guard.cli.serve_llm \
  --model_path "$INPUT_MODEL_PATH" \
  --host "$HOST" \
  --port "$INPUT_PORT" \
  --workers "$WORKERS" \
  --mode "$MODE" \
  --request-timeout "$REQUEST_TIMEOUT" \
  --service input \
  "${gpu_args[@]}" \
  "$@"
