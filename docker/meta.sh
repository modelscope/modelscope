#!/bin/bash
apt-get update && apt-get install -y libsox-dev unzip libaio-dev zip iputils-ping telnet sudo git net-tools && apt-get clean && rm -rf /var/lib/apt/lists/*

mkdir -p ~/miniconda3
rm -rf ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
. ~/miniconda3/bin/activate

pip install --no-cache-dir funtextprocessing typeguard==2.13.3 scikit-learn scipy<1.13.0 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

curl -fsSL https://ollama.com/install.sh | sh

pip install --no-cache-dir -U funasr

pip install --no-cache-dir -U qwen_vl_utils pyav librosa autoawq timm transformers accelerate peft optimum trl safetensors

pip install --no-cache-dir torchsde jupyterlab torchmetrics==0.11.4 tiktoken transformers_stream_generator bitsandbytes basicsr

pip install --no-cache-dir text2sql_lgesql==1.3.0 git+https://github.com/jin-s13/xtcocoapi.git@v1.14 git+https://github.com/gatagat/lap.git@v0.4.0 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --force --no-deps

pip install --no-cache-dir mpi4py paint_ldm mmcls>=0.21.0 mmdet>=2.25.0 decord>=0.6.0 ipykernel fasttext fairseq deepspeed apex -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0" pip install --no-cache-dir  'git+https://github.com/facebookresearch/detectron2.git';

cd /tmp && git clone https://github.com/Dao-AILab/flash-attention.git && cd flash-attention && python setup.py install && cd / && rm -fr /tmp/flash-attention && pip cache purge;

pip install --no-cache-dir auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

pip install --no-cache-dir --force tinycudann==1.7  -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip uninstall -y torch-scatter && TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0" pip install --no-cache-dir -U torch-scatter

pip install --no-cache-dir -U triton

pip install vllm -U

pip install --no-cache-dir -U lmdeploy --no-deps

pip install --no-cache-dir -U torch torchvision torchaudio
