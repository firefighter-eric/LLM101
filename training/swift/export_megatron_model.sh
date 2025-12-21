clear

LD_LIBRARY_PATH=/home/eric/anaconda3/envs/megatron/lib/python3.10/site-packages/torch/lib:\
/home/eric/miniconda3/envs/swift/lib/python3.10/site-packages/torch/lib:\
$LD_LIBRARY_PATH
echo $LD_LIBRARY_PATH

rm -rf data/models/Qwen/Qwen3-0.6B-mcore
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model data/models/Qwen/Qwen3-0.6B \
    --to_mcore true \
    --torch_dtype bfloat16 \
    --output_dir data/models/Qwen/Qwen3-0.6B-mcore
    
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model data/models/Qwen/Qwen3-0.6B \
    --to_mcore true \
    --output_dir data/models/Qwen/Qwen3-0.6B-mcore
    
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --model data/models/Qwen/Qwen3-0.6B-FP8 \
    --to_mcore true \
    --output_dir data/models/Qwen/Qwen3-0.6B-FP8-mcore \
    --test_convert_precision true
