clear

export WANDB_PROJECT=llm101

NPROC_PER_NODE=1 \
CUDA_VISIBLE_DEVICES=0 \
megatron sft \
    --load data/models/Qwen/Qwen3-0.6B-mcore \
    --dataset data/swift/Qwen3-SFT-Mixin/qwen3_32b_distill_1k.jsonl \
    --tensor_model_parallel_size 1 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity selective \
    --train_iters 31 \
    --eval_iters 5 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-5 \
    --lr_warmup_iters 10 \
    --min_lr 1e-6 \
    --save runs/test/qwen3-0.6b-megatron \
    --save_interval 100 \
    --max_length 2048 \
    --system 'You are a helpful assistant.' \
    --num_workers 4 \
    --no_save_optim true \
    --no_save_rng true \
    --dataset_num_proc 4 \
    --bf16 true \
    --log_interval 1 \
    --use_flash_attn true