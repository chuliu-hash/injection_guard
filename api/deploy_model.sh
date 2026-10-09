#!/bin/bash
# 请在项目model-shield-system的根目录下运行 参考 cd /root/autodl-tmp/model-shield-system 
source /root/miniconda3/etc/profile.d/conda.sh
conda activate LLM_Guards

cd pipelines

python app_pj.py  &

cd user/Prompt_Injection 


# # ----------  后台启动两个服务 ----------
echo "启动模型 1 (Qwen_prompt_injection)，端口 8094 ..."
python interact_pj.py \
    --model_path ./model/LLMS/Qwen_prompt_injection \
    --port 8094 &

echo "启动模型 2 (Qwen3-8B)，端口 8095 ..."
python interact_pj.py \
    --model_path ./model/LLMS/Qwen3-8B \
    --port 8095 &

# echo "启动模型 3 (Qwen3-Embedding-0.6B)，端口 8096 ..."
# python interact_embed.py \
#     --model_path ./model/LLMS/Qwen3-Embedding-0.6B \
#     --port 8096 &

wait