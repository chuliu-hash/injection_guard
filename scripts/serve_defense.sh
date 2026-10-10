#!/usr/bin/env bash
# 启动防御 API（FastAPI :DEFENSE_PORT，默认 8099）。
#
# 按 configs/deploy.env 的端口派生并注入 APP_PORT 与各 *_API_URL，与
# `injection-guard-deploy` 的行为一致（单独手动起各服务时用本脚本）。
# 额外参数透传，如：scripts/serve_defense.sh --help
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

export APP_PORT="$DEFENSE_PORT"
export INPUT_DETECTOR_API_URL="http://${URL_HOST}:${INPUT_PORT}/v1"
export OUTPUT_DETECTOR_API_URL="http://${URL_HOST}:${OUTPUT_PORT}/v1"
# 仅在本机部署对应服务时覆盖，避免盖掉 .env 里配置的外部 Embedding / 云端聊天连接
if [[ "$DEPLOY_LOCAL_EMBED" == "true" ]]; then
  export EMBED_API_URL="http://${URL_HOST}:${EMBED_PORT}/v1"
fi
if [[ "$DEPLOY_LOCAL_CHAT" == "true" ]]; then
  export CHAT_API_URL="http://${URL_HOST}:${CHAT_PORT}/v1"
fi

exec "$PYTHON_BIN" -m injection_guard.cli.serve_defense "$@"
