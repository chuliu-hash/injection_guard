#!/bin/bash
# 一键启动防御 API 与三个 LLM 推理服务（输入检测 / 输出检测 / 普通聊天）。
# 前置：在项目根目录 `pip install -e .`，并把模型权重放到 $MODEL_ROOT 下。
#
# 前端：本脚本**不**自动构建前端（避免拖慢模型启动）。若需防御 API 在 `/` 托管 Web 界面，
# 请先手动构建一次：`cd frontend && npm ci && npm run build`（产出 frontend/dist）。
#
# 用法：MODEL_ROOT=/path/to/models bash scripts/deploy_model.sh
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate LLM_Guards

MODEL_ROOT="${MODEL_ROOT:-./models}"
LLMS_DIR="$MODEL_ROOT/Prompt_Injection/model/LLMS"

echo "启动防御 API，端口 8099 ..."
python -m injection_guard.cli.serve_defense &

echo "启动模型 1 (Qwen_prompt_injection)，端口 8094 ..."
python -m injection_guard.cli.serve_llm \
    --model_path "$LLMS_DIR/Qwen_prompt_injection" \
    --port 8094 &

echo "启动模型 2 (Qwen3-8B)，端口 8095 ..."
python -m injection_guard.cli.serve_llm \
    --model_path "$LLMS_DIR/Qwen3-8B" \
    --port 8095 &

# 普通聊天模型（用于生成 model_response），端口 8097；按需改成你的通用对话模型路径
echo "启动聊天模型，端口 8097 ..."
python -m injection_guard.cli.serve_llm \
    --model_path "${CHAT_MODEL_PATH:-$LLMS_DIR/Qwen3-8B}" \
    --port 8097 &

# Embedding 服务（RAG 增强所需，启用 ENABLE_RAG=true 时启动），端口 8096
# python -m injection_guard.cli.serve_embed \
#     --model_path "$LLMS_DIR/Qwen3-Embedding-0.6B" \
#     --port 8096 &

wait
