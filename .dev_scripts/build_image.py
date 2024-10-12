import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str)
parser.add_argument('--torch_version', type=str)
parser.add_argument('--torchvision_version', type=str)
parser.add_argument('--torchaudio_version', type=str)
parser.add_argument('--modelscope_branch', type=str)
parser.add_argument('--swift_branch', type=str)


args = parser.parse_args()

content = f"""
FROM {args.base_image}

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    cp /tmp/resources/ubuntu2204.aliyun /etc/apt/sources.list 

#content#

#mscontent#

ENV SETUPTOOLS_USE_DISTUTILS=stdlib
ENV VLLM_USE_MODELSCOPE=True
ENV LMDEPLOY_USE_MODELSCOPE=True 
ENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope
"""

inner_content = ''
with open('../docker/meta.sh', 'r') as f:
    for line in f.readlines():
        if line.strip():
            inner_content += 'RUN ' + line

inner_content = inner_content.replace('$torch_version', args.torch_version)
inner_content = inner_content.replace('$torchvision_version', args.torchvision_version)
inner_content = inner_content.replace('$torchaudio_version', args.torchaudio_version)
content = content.replace('#content#', inner_content)

inner_content = ''
with open('../docker/meta_ms.sh', 'r') as f:
    for line in f.readlines():
        if line.strip():
            inner_content += 'RUN ' + line


inner_content = inner_content.replace('$modelscope_branch', args.modelscope_branch)
inner_content = inner_content.replace('$swift_branch', args.swift_branch)
content = content.replace('#mscontent#', inner_content)

with open('../Dockerfile', 'w') as f:
    f.write(content)

