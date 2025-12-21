echo CUDA_HOME=${CUDA_HOME}
conda create -n swift311cu128 python=3.12 -y
conda activate swift311cu128

uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install ms-swift==3.11 pybind11 ninja wandb

# Fix for missing nccl.h and libnccl.so during transformer_engine build
export NCCL_ROOT=$(python -c "import nvidia.nccl; print(nvidia.nccl.__path__[0])")
export CPATH=$NCCL_ROOT/include:$CPATH
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH
echo NCCL_ROOT=${NCCL_ROOT}

# transformer_engine
# 若出现安装错误，可以参考该issue解决: https://github.com/modelscope/ms-swift/issues/3793
uv pip install --no-build-isolation transformer_engine[pytorch]==2.10
# 或使用以下方式安装
pip install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@release_v2.5#egg=transformer_engine[pytorch]

# apex
# 提示：Megatron-SWIFT可以在不含apex的环境下运行，额外设置`--no_gradient_accumulation_fusion true`即可。
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# megatron-core
uv pip install git+https://github.com/NVIDIA/Megatron-LM.git@core_r0.14.0

# 若使用多机训练，请额外设置`MODELSCOPE_CACHE`环境变量为共享存储路径
# 这将确保数据集缓存共享，而加速预处理速度。
# 注意：这步很关键，不然多机训练可能因随机性问题导致数据不一致而训练卡住。
export MODELSCOPE_CACHE='/xxx/shared'

# Megatron-LM
# 依赖库Megatron-LM中的训练模块将由swift进行git clone并安装。你也可以通过环境变量`MEGATRON_LM_PATH`指向已经下载好的repo路径（断网环境，[core_r0.14.0分支](https://github.com/NVIDIA/Megatron-LM/tree/core_r0.14.0)）。
git clone --branch core_r0.14.0 https://github.com/NVIDIA/Megatron-LM.git
export MEGATRON_LM_PATH='/xxx/Megatron-LM'

# flash_attn
# 选择合适的版本进行安装：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3
# 注意：请勿安装高于transformer_engine限制的最高版本：https://github.com/NVIDIA/TransformerEngine/blob/release_v2.10/transformer_engine/pytorch/attention/dot_product_attention/utils.py#L109
MAX_JOBS=8 uv pip install "flash-attn==2.8.3" --no-build-isolation
uv pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"