# syntax = docker/dockerfile:experimental
#
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference:
#           https://docs.docker.com/develop/develop-images/build_enhancements/

# ARG BASE_IMAGE=reg.docker.alibaba-inc.com/pai-dlc/pytorch-training:1.10PAI-gpu-py36-cu113-ubuntu18.04
# FROM ${BASE_IMAGE} as dev-base

# FROM reg.docker.alibaba-inc.com/pai-dlc/pytorch-training:1.10PAI-gpu-py36-cu113-ubuntu18.04 as dev-base
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-devel
# FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
# config pip source
RUN mkdir /root/.pip
COPY docker/rcfiles/pip.conf.tsinghua  /root/.pip/pip.conf
COPY docker/rcfiles/sources.list.aliyun /etc/apt/sources.list

# Install essential Ubuntu packages
RUN apt-get update &&\
    apt-get install -y software-properties-common \
    build-essential \
    git \
    wget \
    vim \
    curl \
    zip \
    zlib1g-dev \
    unzip \
    pkg-config

# install modelscope and its python env
WORKDIR /opt/modelscope
COPY . .
RUN pip install -r requirements.txt
# RUN --mount=type=cache,target=/opt/ccache \
#     python setup.py install

# opencv-python-headless conflict with opencv-python installed
RUN python setup.py install \
    && pip uninstall -y opencv-python-headless

# prepare modelscope libs
COPY docker/scripts/install_libs.sh /tmp/
RUN bash /tmp/install_libs.sh && \
    rm -rf /tmp/install_libs.sh

ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/modelscope/lib64

WORKDIR /workspace
