clear

torchrun --standalone --nnodes=1 --nproc_per_node=1 \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=data/gsm8k/train.parquet \
    data.val_files=data/gsm8k/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-5 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size=1 \
    data.micro_batch_size_per_gpu=1 \
    data.train_batch_size=16 \
    data.max_length=2048 \
    model.partial_pretrain=data/models/Qwen/Qwen2.5-0.5B-Instruct \
    model.use_liger=true \
    model.enable_gradient_checkpointing=true \
    trainer.default_local_dir=runs/verl/qwen2.5-0.5b \
    trainer.project_name=llm101 \
    trainer.experiment_name=gsm8k-sft-qwen-2.5-0.5b-instruct-sp1 \
    trainer.logger=['console','wandb'] \
    trainer.total_training_steps=31 \
    trainer.default_hdfs_dir=null $@ \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true