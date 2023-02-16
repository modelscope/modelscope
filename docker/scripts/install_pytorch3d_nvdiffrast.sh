export CMAKE_BUILD_PARALLEL_LEVEL=36 && export MAX_JOBS=36 && export CMAKE_CUDA_ARCHITECTURES="50;52;60;61;70;75;80;86" \
	&& pip install --no-cache-dir fvcore iopath \
	&& curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz \
        && tar xzf 1.10.0.tar.gz \
        && export CUB_HOME=$PWD/cub-1.10.0 \
        && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" \
	&& rm -fr 1.10.0.tar.gz cub-1.10.0 \
        && apt-get update \
	&& apt-get install -y --no-install-recommends pkg-config libglvnd0 libgl1 libglx0 libegl1  libgles2 libglvnd-dev libgl1-mesa-dev libegl1-mesa-dev  libgles2-mesa-dev -y \
        && git clone https://github.com/NVlabs/nvdiffrast.git \
	&& cd nvdiffrast \
        && pip install --no-cache-dir . \
        && cd .. \
        && rm -rf nvdiffrast
