#!/usr/bin/env bash
# 一键启动全部服务（= injection-guard-deploy）：读 configs/deploy.env，按端口派生各
# *_API_URL 并以子进程拉起 defense / input / output（+ embed / chat 视开关）。
#
# 常用：
#   scripts/serve_all.sh                 # 启动默认服务集
#   scripts/serve_all.sh --dry-run       # 只打印派生环境变量与命令
#   scripts/serve_all.sh --only input,output
#   scripts/serve_all.sh --host 0.0.0.0 --url-host 10.0.0.5
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

exec "$PYTHON_BIN" -m injection_guard.cli.deploy "$@"
