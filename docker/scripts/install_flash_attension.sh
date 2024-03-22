 git clone -b v2.3.3 https://github.com/Dao-AILab/flash-attention && \
    cd flash-attention && MAX_JOBS=46 python setup.py install && \
    cd .. && \
    rm -rf flash-attention
