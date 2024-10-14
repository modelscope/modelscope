import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str)
parser.add_argument('--image_type', type=str)
parser.add_argument('--torch_version', type=str)
parser.add_argument('--torchvision_version', type=str)
parser.add_argument('--cuda_version', type=str)
parser.add_argument('--torchaudio_version', type=str)
parser.add_argument('--tf_version', type=str)
parser.add_argument('--vllm_version', type=str)
parser.add_argument('--lmdeploy_version', type=str)
parser.add_argument('--autogptq_version', type=str)
parser.add_argument('--modelscope_branch', type=str)
parser.add_argument('--swift_branch', type=str)


args = parser.parse_args()

assert args.modelscope_branch
assert args.swift_branch

extra_content = """
pip install adaseq
pip install pai-easycv
"""

if args.image_type == 'cpu':
    if not args.base_image:
        args.base_image = f'reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-py310-torch{args.torch_version}-base'
    if not args.torch_version:
        args.torch_version = '2.3.0'
        args.torchaudio_version = '2.3.0'
        args.torchvision_version = '0.18.0'
    if not args.cuda_version:
        args.cuda_version = '12.1.0'
    args.tf_version = None
    meta_file = '../docker/install_cpu.sh'
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.modelscope_branch} {args.swift_branch}'
elif args.image_type == 'gpu':
    if not args.base_image:
        args.base_image = f'reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda{args.cuda_version}-py310-torch{args.torch_version}-tf{args.tf_version}-base'
    if not args.torch_version:
        args.torch_version = '2.3.0'
        args.torchaudio_version = '2.3.0'
        args.torchvision_version = '0.18.0'
    if not args.tf_version:
        args.tf_version = '1.15.5'
    if not args.cuda_version:
        args.cuda_version = '12.1.0'
    meta_file = '../docker/install.sh'
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.vllm_version} {args.lmdeploy_version} {args.autogptq_version} {args.modelscope_branch} {args.swift_branch}'
elif args.image_type == 'llm':
    if not args.base_image:
        args.base_image = 'pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel'
    if not args.torch_version:
        args.torch_version = '2.4.0'
        args.torchaudio_version = '2.4.0'
        args.torchvision_version = '0.19.0'
    if not args.cuda_version:
        args.cuda_version = '12.4.0'
    args.tf_version = None
    meta_file = '../docker/install.sh'
    extra_content = ''
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.vllm_version} {args.lmdeploy_version} {args.autogptq_version} {args.modelscope_branch} {args.swift_branch}'
else:
    raise ValueError(f'Image type not supported: {args.image_type}')

content = f"""
FROM {args.base_image}

RUN apt-get update && \
    apt-get install -y libsox-dev unzip libaio-dev zip iputils-ping telnet sudo git net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    cp /tmp/resources/ubuntu2204.aliyun /etc/apt/sources.list 

mkdir -p ~/miniconda3
rm -rf ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
. ~/miniconda3/bin/activate

{extra_content}

COPY {meta_file} /tmp/install.sh

RUN sh install.sh {version_args}

ENV SETUPTOOLS_USE_DISTUTILS=stdlib
ENV VLLM_USE_MODELSCOPE=True
ENV LMDEPLOY_USE_MODELSCOPE=True 
ENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope
"""

with open('../Dockerfile', 'w') as f:
    f.write(content)
