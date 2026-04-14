FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV arch=x86_64

COPY docker/scripts/modelscope_env_init.sh /usr/local/bin/ms_env_init.sh
RUN apt-get update && \
    apt-get install -y libsox-dev unzip libaio-dev zip iputils-ping telnet sudo git net-tools zstd libzstd-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV TZ=Asia/Shanghai
ENV arch=x86_64
SHELL ["/bin/bash", "-c"]
COPY docker/rcfiles /tmp/resources
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
RUN wget -O /tmp/boost.tar.gz https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz && \
    cd /tmp && tar xzf boost.tar.gz  && \
    mv /tmp/boost_1_80_0/boost /usr/include/boost && \
    rm -rf /tmp/boost_1_80_0 && rm -rf boost.tar.gz

#install and config python copy from https://github.com/docker-library/python/blob/1b7a1106674a21e699b155cbd53bf39387284cca/3.10/bookworm/Dockerfile
ARG PYTHON_VERSION=3.11.11
ENV PATH /usr/local/bin:$PATH
ENV GPG_KEY="A035C8C19219BA821ECEA86B64E628F8D684696D 7169605F62C751356D054A26A821E680E5FA6305"
ENV PYTHON_VERSION 3.11.11

#install and config python copy from https://github.com/docker-library/python/blob/1b7a1106674a21e699b155cbd53bf39387284cca/3.10/bookworm/Dockerfile
ARG PYTHON_VERSION=3.11.11
ENV PATH /usr/local/bin:$PATH
ENV GPG_KEY="A035C8C19219BA821ECEA86B64E628F8D684696D 7169605F62C751356D054A26A821E680E5FA6305"
ENV PYTHON_VERSION 3.11.11

RUN set -eux; \
        \
        wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
        wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
        GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
        gpg --batch --keyserver hkp://keyserver.ubuntu.com --recv-keys $GPG_KEY; \
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
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:-}" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
        ; \
        rm python; \
        make -j "$nproc" \
                "EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
                "LDFLAGS=${LDFLAGS:--Wl},-rpath='\$\$ORIGIN/../lib'" \
                "PROFILE_TASK=${PROFILE_TASK:-}" \
                python \
        ; \
        make install; \
        \
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
                \) -exec rm -rf '{}' + \
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
# pip>=23.3 required for Python 3.12 (pkgutil.ImpImporter removed; older pip crashes on any install)
ENV PYTHON_PIP_VERSION 24.3.1
# https://github.com/docker-library/python/issues/365
ENV PYTHON_SETUPTOOLS_VERSION 75.8.2
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

RUN pip install --no-cache-dir -U icecream soundfile pybind11 py-spy


COPY ./docker/install.sh /tmp/install.sh

ARG INSTALL_MS_DEPS=True

ARG IMAGE_TYPE=gpu

# install dependencies
COPY requirements /var/modelscope

RUN pip uninstall ms-swift modelscope -y && pip --no-cache-dir install pip==23.* -U && \
if [ "$INSTALL_MS_DEPS" = "True" ]; then \
    pip --no-cache-dir install omegaconf==2.0.6 && \
    pip install 'editdistance==0.8.1' && \
    pip install --no-cache-dir 'cython<=0.29.36' versioneer 'numpy<2.0' && \
    pip install --no-cache-dir -r /var/modelscope/framework.txt && \
    pip install --no-cache-dir -r /var/modelscope/audio.txt -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html && \
    pip install --no-cache-dir -r /var/modelscope/tests.txt && \
    pip install --no-cache-dir -r /var/modelscope/server.txt && \
    pip install --no-cache-dir https://modelscope.oss-cn-beijing.aliyuncs.com/packages/imageio_ffmpeg-0.4.9-py3-none-any.whl --no-dependencies --force && \
    pip install --no-cache-dir 'scipy<1.13.0' && \
    pip install --no-cache-dir typeguard==2.13.3 scikit-learn -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html && \
    pip install --no-cache-dir decord>=0.6.0 mpi4py paint_ldm ipykernel fasttext -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html && \
    pip install --no-cache-dir 'blobfile>=1.0.5' && \
    pip uninstall MinDAEC -y && \
    pip install https://modelscope.oss-cn-beijing.aliyuncs.com/releases/dependencies/MinDAEC-0.0.2-py3-none-any.whl && \
    pip cache purge; \
else \
    pip install --no-cache-dir -r /var/modelscope/framework.txt && \
    pip cache purge; \
fi

ARG CUR_TIME=20260414175909
RUN echo $CUR_TIME

RUN bash /tmp/install.sh 2.9.1 0.24.1 2.9.1 0.15.1 0.11.0 0.7.1 2.8.3 2.0.0 && \
    pip install --no-cache-dir -U funasr scikit-learn && \
    pip install --no-cache-dir -U qwen_vl_utils qwen_omni_utils librosa timm transformers accelerate peft trl safetensors && \
    cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b main  --single-branch https://github.com/modelscope/ms-swift.git && \
    cd ms-swift && pip install .[llm] && \
    pip install .[eval] && pip install evalscope -U --no-dependencies && pip install ms-agent -U --no-dependencies && \
    cd / && rm -fr /tmp/ms-swift && pip cache purge; \
    cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone -b  master  --single-branch https://github.com/modelscope/modelscope.git && \
    cd modelscope && pip install . -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html && \
    cd / && rm -fr /tmp/modelscope && pip cache purge; \
    pip install --no-cache-dir torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1  && \
    pip install --no-cache-dir transformers diffusers timm>=0.9.0 && pip cache purge; \
    pip install --no-cache-dir omegaconf==2.3.0 && pip cache purge; \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set install.trusted-host mirrors.aliyun.com && \
    cp /tmp/resources/ubuntu2204.aliyun /etc/apt/sources.list


RUN if [ "$IMAGE_TYPE" = "gpu" ]; then \
    pip install --no-cache-dir math_verify "gradio<5.33" "deepspeed<0.18" ray -U && \
    pip install --no-cache-dir liger_kernel wandb swanlab nvitop pre-commit "transformers" "trl<0.25" "peft<0.18" huggingface-hub -U && \
    pip install --no-cache-dir --no-build-isolation transformer_engine[pytorch]; \
    cd /tmp && GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/NVIDIA/apex && \
    cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && \
    cd / && rm -fr /tmp/apex && pip cache purge; \
    pip install --no-cache-dir "megatron-core==0.15.*"; \
    pip uninstall autoawq -y; \
elif [ "$IMAGE_TYPE" = "cpu" ]; then \
    pip install --no-cache-dir huggingface-hub transformers peft diffusers -U; \
else \
    pip install "transformers<5.0" "tokenizers<0.22" "trl<0.23" "diffusers<0.35" --no-dependencies; \
fi

# install nvm and set node version to 18
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash && \
    . $NVM_DIR/nvm.sh && \
    nvm install 22 && \
    nvm use 22

RUN rm -f /etc/apt/sources.list.d/cuda-*.list && apt-get update

ENV VLLM_USE_MODELSCOPE=True
ENV LMDEPLOY_USE_MODELSCOPE=True
ENV MODELSCOPE_CACHE=/mnt/workspace/.cache/modelscope/hub
SHELL ["/bin/bash", "-c"]
