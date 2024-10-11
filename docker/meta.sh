
apt-get update && \
    apt-get install -y libsox-dev unzip libaio-dev zip iputils-ping telnet sudo && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir numpy 'cython<=0.29.36' funtextprocessing kwsbp==0.0.6 safetensors typeguard==2.13.3 scikit-learn librosa==0.9.2 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install torch=$TORCH_VERSION torchvision=$TORCHVISION_VERSION torchaudio=$TORCHAUDIO_VERSION

curl -fsSL https://ollama.com/install.sh | sh

pip install --no-cache-dir -U adaseq pai-easycv &&  pip install --no-cache-dir -U 'ms-swift' 'decord' 'qwen_vl_utils' 'pyav' 'librosa' 'funasr' autoawq 'timm>0.9.5' 'transformers' 'accelerate' 'peft' 'optimum' 'trl'

pip install --no-cache-dir adaseq text2sql_lgesql==1.3.0 git+https://github.com/jin-s13/xtcocoapi.git@v1.14 git+https://github.com/gatagat/lap.git@v0.4.0 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html --force --no-deps

pip install --no-cache-dir mpi4py paint_ldm mmcls>=0.21.0 mmdet>=2.25.0 decord>=0.6.0 ipykernel fasttext fairseq deepspeed apex -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6 8.9 9.0" pip install --no-cache-dir  'git+https://github.com/facebookresearch/detectron2.git';

pip install --no-cache-dir torchsde jupyterlab torchmetrics==0.11.4 tiktoken transformers_stream_generator bitsandbytes basicsr optimum

pip install --no-cache-dir flash_attn==2.5.9.post1 -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip install --no-cache-dir auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu121/

pip install --no-cache-dir -U 'xformers<0.0.27' --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir --force tinycudann==1.7  -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip uninstall -y torch-scatter && TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0" pip install --no-cache-dir -U torch-scatter

pip install --no-cache-dir -U triton https://modelscope.oss-cn-beijing.aliyuncs.com/packages/lmdeploy-0.5.0-cp310-cp310-linux_x86_64.whl

pip install vllm -U

pip install lmdeploy -U
