conda create -n verl python=3.10 -y
conda activate verl
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
pip install liger-kernel

# verl
git clone https://github.com/volcengine/verl.git
cd verl
pip install -e .[vllm]

# apex
pip3 install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" git+https://github.com/NVIDIA/apex

# transformer engine
pip install --no-build-isolation transformer_engine[pytorch]
pip install --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@v2.2
# megatron core
pip install --no-deps megatron-core==0.12.0