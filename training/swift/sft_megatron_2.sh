clear

set -x
export MODELSCOPE_CACHE='/home/eric/.cache/shared'
export MEGATRON_LM_PATH='/home/eric/projects/Megatron-LM'

export WANDB_PROJECT=llm101

NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron sft \
    --model Qwen/Qwen3-0.6B \
    --load_safetensors true \
    --save_safetensors true \
    --dataset swift/Qwen3-SFT-Mixin \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --no_gradient_accumulation_fusion true \
    --recompute_granularity selective \
    --max_epochs 3 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 10 \
    --min_lr 1e-6 \
    --save_interval 100 \
    --max_length 2048 \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --bf16 true \
    --log_interval 1 \
    --use_flash_attn true \
    --wandb_project llm101 \
    --wandb_exp_name swift311-qwen3-0.6b-2