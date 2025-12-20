clear

export WANDB_PROJECT=llm101

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model data/models/Qwen/Qwen3-0.6B \
    --train_type full \
    --dataset data/swift/Qwen3-SFT-Mixin/qwen3_32b_distill_1k.jsonl \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_length 2048 \
    --output_dir runs/test/qwen3-0.6b-trl \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --attn_impl flash_attention_2 \
    --use_liger_kernel true \
    --run_name test-swift-qwen3-0.6b-trl
    # --tuner_backend unsloth