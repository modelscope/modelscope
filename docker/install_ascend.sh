#!/bin/bash

torch_version=${1:-2.7.1}
torchvision_version=${2:-0.22.1}
torch_npu_version=${3:-2.7.1}
vllm_version=${4:-0.11.0}
vllm_ascend_version=${5:-0.11.0-dev}


pip uninstall -y torch torchvision torchaudio

pip install --no-cache-dir torch==$torch_version torchvision==$torchvision_version torch-npu=$torch_npu_version

pip install --no-cache-dir tiktoken transformers_stream_generator bitsandbytes deepspeed torchmetrics decord optimum openai-whisper

cd /usr/local/Ascend/
git clone https://github.com/vllm-project/vllm.git && cd vllm
git checkout releases/v$vllm_version
VLLM_TARGET_DEVICE=empty pip install -e .

cd /usr/local/Ascend/
git clone https://github.com/vllm-project/vllm-ascend.git && cd vllm-ascend
git checkout v$vllm_ascend_version
PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi pip install . --trusted-host mirrors.huaweicloud.com