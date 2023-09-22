    git clone -b v1.0.8 https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && pip install . && \
    pip install csrc/layer_norm && \
    pip install csrc/rotary && \
    cd .. && \
    rm -rf flash-attention
