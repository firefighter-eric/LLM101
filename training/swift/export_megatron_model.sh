# LD_LIBRARY_PATH=/home/eric/anaconda3/envs/megatron/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH \

export MODELSCOPE_CACHE='/home/eric/.cache/shared'
export MEGATRON_LM_PATH='/home/eric/projects/Megatron-LM'

CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model data/models/Qwen/Qwen3-0.6B \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir data/models/Qwen/Qwen3-0.6B-mcore-2