#!/bin/bash

torch_version=${1:-2.4.0}
torchvision_version=${2:-0.19.0}
torchaudio_version=${3:-2.4.0}
vllm_version=${4:-0.6.0}
lmdeploy_version=${5:-0.6.1}
autogptq_version=${6:-0.7.1}

pip install --no-cache-dir -U autoawq

pip uninstall -y torch torchvision torchaudio

pip install --no-cache-dir torch==$torch_version torchvision==$torchvision_version torchaudio==$torchaudio_version

pip install --no-cache-dir tiktoken transformers_stream_generator bitsandbytes deepspeed torchmetrics decord optimum

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# find on: https://github.com/Dao-AILab/flash-attention/releases
cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && python setup.py install && cd / && rm -fr /tmp/flash-attention && pip cache purge;

pip install --no-cache-dir auto-gptq==$autogptq_version

# pip uninstall -y torch-scatter && TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0" pip install --no-cache-dir -U torch-scatter

pip install --no-cache-dir -U triton

pip install --no-cache-dir vllm==$vllm_version -U

pip install --no-cache-dir -U lmdeploy==$lmdeploy_version --no-deps
