#!/bin/bash

set -eo pipefail

ModelScopeLib=/usr/local/modelscope/lib64

if [ ! -d /usr/local/modelscope ]; then
    mkdir -p $ModelScopeLib
fi

# audio libs
wget "http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/maas/libs/audio/libmitaec_pyio.so" -O ${ModelScopeLib}/libmitaec_pyio.so
