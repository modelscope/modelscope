#!/bin/bash

torch_version=${1:-2.4.0}
torchvision_version=${2:-0.19.0}
torchaudio_version=${3:-2.4.0}

pip uninstall -y torch torchvision torchaudio

pip install --no-cache-dir torch==$torch_version torchvision==$torchvision_version torchaudio==$torchaudio_version --index-url https://download.pytorch.org/whl/cpu
