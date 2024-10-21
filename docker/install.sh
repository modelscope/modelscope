#!/bin/bash

torch_version=${1:-2.4.0}
torchvision_version=${2:-0.19.0}
torchaudio_version=${3:-2.4.0}
vllm_version=${4:-0.6.0}
lmdeploy_version=${5:-0.6.1}
autogptq_version=${6:-0.7.1}
modelscope_branch=${7:-master}
swift_branch=${8:-main}

pip install --no-cache-dir funtextprocessing typeguard==2.13.3 scikit-learn -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# curl -fsSL https://ollama.com/install.sh | sh

pip install --no-cache-dir -U funasr

pip install --no-cache-dir -U qwen_vl_utils pyav librosa autoawq timm transformers accelerate peft optimum trl safetensors

pip install --no-cache-dir torchsde jupyterlab torchmetrics==0.11.4 tiktoken transformers_stream_generator bitsandbytes basicsr

pip install --no-cache-dir text2sql_lgesql==1.3.0 git+https://github.com/jin-s13/xtcocoapi.git@v1.14 git+https://github.com/gatagat/lap.git@v0.4.0 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --force --no-deps

pip install --no-cache-dir mpi4py paint_ldm mmcls>=0.21.0 mmdet>=2.25.0 decord>=0.6.0 ipykernel fasttext fairseq deepspeed apex -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0" pip install --no-cache-dir  'git+https://github.com/facebookresearch/detectron2.git';

# pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# find on: https://github.com/Dao-AILab/flash-attention/releases
# cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && python setup.py install && cd / && rm -fr /tmp/flash-attention && pip cache purge;

pip install --no-cache-dir auto-gptq==$autogptq_version

pip install --no-cache-dir --force tinycudann==1.7  -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

# pip uninstall -y torch-scatter && TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0" pip install --no-cache-dir -U torch-scatter

pip install --no-cache-dir -U triton

pip install vllm==$vllm_version -U

pip install --no-cache-dir -U lmdeploy==$lmdeploy_version --no-deps

pip install --no-cache-dir -U torch==$torch_version torchvision==$torchvision_version torchaudio==$torchaudio_version

pip uninstall ms-swift modelscope -y

cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $modelscope_branch  --single-branch https://github.com/modelscope/modelscope.git && cd modelscope && pip install .[all] && cd / && rm -fr /tmp/modelscope && pip cache purge;

cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $swift_branch  --single-branch https://github.com/modelscope/ms-swift.git && cd ms-swift && pip install .[all] && cd / && rm -fr /tmp/ms-swift && pip cache purge;
