#!/usr/bin/env bash
# 各服务启动脚本的公共部分：定位仓库根、激活 conda 环境、载入 configs/deploy.env。
#
# 由 serve_*.sh 通过 `source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"` 引入，
# **不要直接执行**。引入后可用变量：$PYTHON_BIN $HOST $URL_HOST $WORKERS $MODE
# $REQUEST_TIMEOUT $GPU_IDS $gpu_args 及各端口 / 模型路径。
#
# 环境变量都可覆盖，例如：
#   GPU_IDS=0,1 scripts/serve_input.sh
#   CONDA_ENV= scripts/serve_input.sh            # 跳过 conda 激活（自行保证 python 就绪）
#   PYTHON=/path/to/python scripts/serve_input.sh
set -euo pipefail

# ── 仓库根（scripts/ 的上一级）──
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"   # .env 相对 CWD 读取，必须从仓库根运行

# ── conda 环境（默认 injection_guard；CONDA_ENV= 置空可跳过）──
CONDA_ENV="${CONDA_ENV-injection_guard}"
if [[ -n "$CONDA_ENV" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi
PYTHON_BIN="${PYTHON:-python}"

# ── 载入 configs/deploy.env（端口 / 模型路径 / workers 的唯一来源）──
# 逐行解析并**只填补未设置的变量** —— 与 DeploySettings 的优先级一致：
# os.environ > configs/deploy.env > 默认值（下面 ${VAR:-默认}）。故 `GPU_IDS=0 scripts/serve_input.sh`
# 这类临时覆盖有效，而非被文件里的值盖掉。
DEPLOY_ENV_FILE="$REPO_ROOT/configs/deploy.env"
if [[ -f "$DEPLOY_ENV_FILE" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    _line="${_line#"${_line%%[![:space:]]*}"}"   # 去行首空白
    [[ -z "$_line" || "$_line" == \#* ]] && continue
    _key="${_line%%=*}"
    [[ "$_key" == "$_line" ]] && continue        # 非 KEY=VALUE 行（无 =）跳过
    _key="${_key%"${_key##*[![:space:]]}"}"      # 去 key 尾部空白
    _val="${_line#*=}"
    [[ -v "$_key" ]] && continue                 # 环境变量已设置 → 优先，不覆盖
    export "$_key=$_val"
  done < "$DEPLOY_ENV_FILE"
  unset _line _key _val
fi

# ── 缺省值（与 configs/deploy.env.example 一致）──
DEFENSE_PORT="${DEFENSE_PORT:-8099}"
INPUT_PORT="${INPUT_PORT:-8094}"
OUTPUT_PORT="${OUTPUT_PORT:-8095}"
EMBED_PORT="${EMBED_PORT:-8096}"
CHAT_PORT="${CHAT_PORT:-8097}"

HOST="${HOST:-localhost}"           # 子服务绑定地址
URL_HOST="${URL_HOST:-localhost}"   # 派生 *_API_URL 用的主机名（勿用 0.0.0.0）

WORKERS="${WORKERS:-1}"
MODE="${MODE:-full}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
GPU_IDS="${GPU_IDS:-}"

INPUT_MODEL_PATH="${INPUT_MODEL_PATH:-./models/LLM/Qwen_prompt_injection}"
OUTPUT_MODEL_PATH="${OUTPUT_MODEL_PATH:-./models/LLM/qwen3-0.6b}"
EMBED_MODEL_PATH="${EMBED_MODEL_PATH:-./models/Embedding/Qwen3-Embedding-0.6B}"
CHAT_MODEL_PATH="${CHAT_MODEL_PATH:-./models/LLM/qwen3-0.6b}"

DEPLOY_LOCAL_EMBED="${DEPLOY_LOCAL_EMBED:-true}"
DEPLOY_LOCAL_CHAT="${DEPLOY_LOCAL_CHAT:-false}"

# GPU_IDS 非空时给 LLM / Embedding 服务追加 --gpu-ids
gpu_args=()
if [[ -n "$GPU_IDS" ]]; then
  gpu_args=(--gpu-ids "$GPU_IDS")
fi
