export CMAKE_BUILD_PARALLEL_LEVEL=36 \
        && export MAX_JOBS=36 \
        && export CMAKE_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86" \
        && git clone --branch 2.1.0 --recursive https://github.com/NVIDIA/thrust.git \
        && cd thrust \
        && mkdir build \
        && cd build \
        && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/cuda/ -DTHRUST_INCLUDE_CUB_CMAKE=ON .. \
        && make install \
        && cd ../.. \
        && rm -rf thrust \
        && pip install --no-cache-dir fvcore iopath \
        && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
        && apt-get update \
        && apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1  libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev  libgles2-mesa-dev -y \
        && git clone https://github.com/NVlabs/nvdiffrast.git \
        && cd nvdiffrast \
        && pip install --no-cache-dir . \
        && cd .. \
        && rm -rf nvdiffrast
