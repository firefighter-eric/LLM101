conda create -n swift python=3.10 -y
pip install uv ninja

# torch
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install flash-attn --no-build-isolation
uv pip install ms-swift liger-kernel


# megatron

# apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex

# transformer engine
uv pip install --no-build-isolation transformer_engine[pytorch]
# pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2

# megatron core
uv pip install --no-deps megatron-core==0.12.0.rc3