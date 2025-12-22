clear

set -x
export CUDA_HOME=/usr/local/cuda-12.9
export MODELSCOPE_CACHE='/home/eric/.cache/shared'
export MEGATRON_LM_PATH='/home/eric/projects/Megatron-LM'

export WANDB_PROJECT=llm101
OUTPUT_NAME=swift311cu129-qwen3-0.6b-5080

NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron sft \
    --train_type full \
    --load data/models/Qwen/Qwen3-0.6B-mcore \
    --save runs/${OUTPUT_NAME} \
    --load_safetensors false \
    --save_safetensors false \
    --dataset data/swift/Qwen3-SFT-Mixin/qwen3_32b_distill_1k.jsonl \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --no_gradient_accumulation_fusion false \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
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
    --attention_backend flash \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_exp_name ${OUTPUT_NAME}

