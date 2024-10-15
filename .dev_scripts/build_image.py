import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_image', type=str, default=None)
parser.add_argument('--image_type', type=str)
parser.add_argument('--python_version', type=str, default='3.10.14')
parser.add_argument('--torch_version', type=str, default=None)
parser.add_argument('--torchvision_version', type=str, default=None)
parser.add_argument('--cuda_version', type=str, default=None)
parser.add_argument('--torchaudio_version', type=str, default=None)
parser.add_argument('--tf_version', type=str, default=None)
parser.add_argument('--vllm_version', type=str, default=None)
parser.add_argument('--lmdeploy_version', type=str, default=None)
parser.add_argument('--autogptq_version', type=str, default=None)
parser.add_argument('--modelscope_branch', type=str, default='master')
parser.add_argument('--swift_branch', type=str, default='main')


args = parser.parse_args()

assert args.modelscope_branch
assert args.swift_branch

if args.image_type == 'cpu':
    if not args.torch_version:
        args.torch_version = '2.3.0'
        args.torchaudio_version = '2.3.0'
        args.torchvision_version = '0.18.0'
    if not args.cuda_version:
        args.cuda_version = '12.1.0'
    args.tf_version = None
    args.vllm_version = None
    args.lmdeploy_version = None
    args.autogptq_version = None
    meta_file = './docker/install_cpu.sh'
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.modelscope_branch} {args.swift_branch}'
    extra_content = """
RUN pip install adaseq
RUN pip install pai-easycv
"""
    if not args.base_image:
        args.base_image = f'reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-py310-torch{args.torch_version}-base'
elif args.image_type == 'gpu':
    if not args.torch_version:
        args.torch_version = '2.3.0'
        args.torchaudio_version = '2.3.0'
        args.torchvision_version = '0.18.0'
    if not args.tf_version:
        args.tf_version = '1.15.5'
    if not args.cuda_version:
        args.cuda_version = '12.1.0'
    if not args.vllm_version:
        args.vllm_version = '0.5.1'
    if not args.lmdeploy_version:
        args.lmdeploy_version = '0.5.0'
    if not args.autogptq_version:
        args.autogptq_version = '0.7.1'
    meta_file = './docker/install.sh'
    extra_content = """
RUN pip install adaseq
RUN pip install pai-easycv
"""
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.vllm_version} {args.lmdeploy_version} {args.autogptq_version} {args.modelscope_branch} {args.swift_branch}'
    if not args.base_image:
        args.base_image = f'reg.docker.alibaba-inc.com/modelscope/modelscope:ubuntu22.04-cuda{args.cuda_version}-py310-torch{args.torch_version}-tf{args.tf_version}-base'
elif args.image_type == 'llm':
    if not args.base_image:
        args.base_image = 'pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel'
    if not args.torch_version:
        args.torch_version = '2.4.0'
        args.torchaudio_version = '2.4.0'
        args.torchvision_version = '0.19.0'
    if not args.cuda_version:
        args.cuda_version = '12.4.0'
    if not args.vllm_version:
        args.vllm_version = '0.6.0'
    if not args.lmdeploy_version:
        args.lmdeploy_version = '0.6.1'
    if not args.autogptq_version:
        args.autogptq_version = '0.7.1'
    args.tf_version = None
    meta_file = './docker/install.sh'
    extra_content = f"""  
ENV TZ=Asia/Shanghai
ENV arch=x86_64
SHELL ["/bin/bash", "-c"]
COPY docker/rcfiles /tmp/resources
COPY docker/jupyter_plugins /tmp/resources/jupyter_plugins
RUN apt-get update && apt-get upgrade -y && apt-get install -y --reinstall ca-certificates && \
    apt-get install -y make apt-utils openssh-server locales wget git strace gdb sox libopenmpi-dev curl \
    iputils-ping net-tools iproute2 autoconf automake gperf libre2-dev libssl-dev \
    libtool libcurl4-openssl-dev libb64-dev libgoogle-perftools-dev patchelf \
    rapidjson-dev scons software-properties-common pkg-config unzip zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev liblzma-dev \
    libarchive-dev libxml2-dev libnuma-dev cmake \
    libgeos-dev strace vim ffmpeg libsm6 tzdata language-pack-zh-hans \
    ttf-wqy-microhei ttf-wqy-zenhei xfonts-wqy libxext6 build-essential ninja-build \
    libjpeg-dev libpng-dev && \
    wget https://packagecloud.io/github/git-lfs/packages/debian/bullseye/git-lfs_3.2.0_amd64.deb/download -O ./git-lfs_3.2.0_amd64.deb && \
    dpkg -i ./git-lfs_3.2.0_amd64.deb && \
    rm -f ./git-lfs_3.2.0_amd64.deb && \
    locale-gen zh_CN && \
    locale-gen zh_CN.utf8 && \
    update-locale LANG=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8 LANGUAGE=zh_CN.UTF-8 && \
    ln -fs /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=zh_CN.UTF-8 LANGUAGE=zh_CN.UTF-8 LC_ALL=zh_CN.UTF-8
RUN wget -O /tmp/boost.tar.gz https://boostorg.jfrog.io/artifactory/main/release/1.80.0/source/boost_1_80_0.tar.gz && \
    cd /tmp && tar xzf boost.tar.gz  && \
    mv /tmp/boost_1_80_0/boost /usr/include/boost && \
    rm -rf /tmp/boost_1_80_0 && rm -rf boost.tar.gz

#install and config python copy from https://github.com/docker-library/python/blob/1b7a1106674a21e699b155cbd53bf39387284cca/3.10/bookworm/Dockerfile
ARG PYTHON_VERSION={args.python_version}
ENV PATH /usr/local/bin:$PATH
ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION {args.python_version}

#install and config python copy from https://github.com/docker-library/python/blob/1b7a1106674a21e699b155cbd53bf39387284cca/3.10/bookworm/Dockerfile
ARG PYTHON_VERSION={args.python_version}
ENV PATH /usr/local/bin:$PATH
ENV GPG_KEY A035C8C19219BA821ECEA86B64E628F8D684696D
ENV PYTHON_VERSION {args.python_version}

RUN set -eux; \
        \
        wget -O python.tar.xz "https://www.python.org/ftp/python/${{PYTHON_VERSION%%[a-z]*}}/Python-$PYTHON_VERSION.tar.xz"; \
        wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${{PYTHON_VERSION%%[a-z]*}}/Python-$PYTHON_VERSION.tar.xz.asc"; \
        GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
        gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
        gpg --batch --verify python.tar.xz.asc python.tar.xz; \
        gpgconf --kill all; \
        rm -rf "$GNUPGHOME" python.tar.xz.asc; \
        mkdir -p /usr/src/python; \
        tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
        rm python.tar.xz; \
        \
        cd /usr/src/python; \
        gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
        ./configure \
                --build="$gnuArch" \
                --enable-loadable-sqlite-extensions \
                --enable-optimizations \
                --enable-option-checking=fatal \
                --enable-shared \
                --with-lto \
                --with-system-expat \
                --without-ensurepip \
        ; \
        nproc="$(nproc)"; \
        EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
        LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${{EXTRA_CFLAGS:-}}" \
                "LDFLAGS=${{LDFLAGS:-}}" \
                "PROFILE_TASK=${{PROFILE_TASK:-}}" \
        ; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
        rm python; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${{EXTRA_CFLAGS:-}}" \
                "LDFLAGS=${{LDFLAGS:--Wl}},-rpath='\$\$ORIGIN/../lib'" \
                "PROFILE_TASK=${{PROFILE_TASK:-}}" \
                python \
        ; \
        make install; \
        \
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
        bin="$(readlink -ve /usr/local/bin/python3)"; \
        dir="$(dirname "$bin")"; \
        mkdir -p "/usr/share/gdb/auto-load/$dir"; \
        cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
        \
        cd /; \
        rm -rf /usr/src/python; \
        \
        find /usr/local -depth \
                \( \
                        \( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
                        -o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
                \) -exec rm -rf '{{}}' + \
        ; \
        \
        ldconfig; \
        \
        python3 --version

# make some useful symlinks that are expected to exist ("/usr/local/bin/python" and friends)
RUN set -eux; \
        for src in idle3 pydoc3 python3 python3-config; do \
                dst="$(echo "$src" | tr -d 3)"; \
                [ -s "/usr/local/bin/$src" ]; \
                [ ! -e "/usr/local/bin/$dst" ]; \
                ln -svT "$src" "/usr/local/bin/$dst"; \
        done

# if this is called "PIP_VERSION", pip explodes with "ValueError: invalid truth value '<VERSION>'"
ENV PYTHON_PIP_VERSION 23.0.1
# https://github.com/docker-library/python/issues/365
ENV PYTHON_SETUPTOOLS_VERSION 65.5.1
# https://github.com/pypa/get-pip
ENV PYTHON_GET_PIP_URL https://github.com/pypa/get-pip/raw/dbf0c85f76fb6e1ab42aa672ffca6f0a675d9ee4/public/get-pip.py
ENV PYTHON_GET_PIP_SHA256 dfe9fd5c28dc98b5ac17979a953ea550cec37ae1b47a5116007395bfacff2ab9

RUN set -eux; \
        \
        wget -O get-pip.py "$PYTHON_GET_PIP_URL"; \
        echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum -c -; \
        \
        export PYTHONDONTWRITEBYTECODE=1; \
        \
        python get-pip.py \
                --disable-pip-version-check \
                --no-cache-dir \
                --no-compile \
                "pip==$PYTHON_PIP_VERSION" \
                "setuptools==$PYTHON_SETUPTOOLS_VERSION" \
        ; \
        rm -f get-pip.py; \
        \
        pip --version
# end of install python
    """
    version_args = f'{args.torch_version} {args.torchvision_version} {args.torchaudio_version} {args.vllm_version} {args.lmdeploy_version} {args.autogptq_version} {args.modelscope_branch} {args.swift_branch}'
else:
    raise ValueError(f'Image type not supported: {args.image_type}')

content = f"""
FROM {args.base_image}

RUN apt-get update && \
    apt-get install -y libsox-dev unzip libaio-dev zip iputils-ping telnet sudo git net-tools && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

{extra_content}

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    cp /tmp/resources/ubuntu2204.aliyun /etc/apt/sources.list 

COPY {meta_file} /tmp/install.sh

RUN sh install.sh {version_args}

ENV SETUPTOOLS_USE_DISTUTILS=stdlib
ENV VLLM_USE_MODELSCOPE=True
ENV LMDEPLOY_USE_MODELSCOPE=True 
ENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope
"""

with open('./Dockerfile', 'w') as f:
    f.write(content)
