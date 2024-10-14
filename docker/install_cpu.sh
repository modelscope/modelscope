#!/bin/bash

torch_version=${1:-2.4.0}
torchvision_version=${2:-0.19.0}
torchaudio_version=${3:-2.4.0}
modelscope_branch=${4:master}
swift_branch=${5:main}

pip install --no-cache-dir funtextprocessing typeguard==2.13.3 scikit-learn -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

curl -fsSL https://ollama.com/install.sh | sh

pip install --no-cache-dir -U funasr

pip install --no-cache-dir -U qwen_vl_utils pyav librosa timm transformers accelerate peft trl safetensors

pip install --no-cache-dir text2sql_lgesql==1.3.0 git+https://github.com/jin-s13/xtcocoapi.git@v1.14 git+https://github.com/gatagat/lap.git@v0.4.0 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --force --no-deps

pip install --no-cache-dir mpi4py paint_ldm mmcls>=0.21.0 mmdet>=2.25.0 decord>=0.6.0 ipykernel fasttext fairseq -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip uninstall torch torchvision torchaudio

pip install --no-cache-dir -U torch==$torch_version torchvision==$torchvision_version torchaudio==$torchaudio_version --index-url https://download.pytorch.org/whl/cpu

pip uninstall ms-swift modelscope

cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $modelscope_branch  --single-branch https://github.com/modelscope/modelscope.git && cd modelscope && pip install .[all] && cd / && rm -fr /tmp/modelscope && pip cache purge;

cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b $swift_branch  --single-branch https://github.com/modelscope/ms-swift.git && cd ms-swift && pip install .[all] && cd / && rm -fr /tmp/ms-swift && pip cache purge;