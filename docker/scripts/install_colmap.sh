wget -q https://cmake.org/files/v3.25/cmake-3.25.2-linux-x86_64.sh \
    && mkdir /opt/cmake \
    && sh cmake-3.25.2-linux-x86_64.sh --prefix=/opt/cmake --skip-license \
    && ln -s /opt/cmake/bin/cmake /usr/local/bin/cmake \
    && rm -f cmake-3.25.2-linux-x86_64.sh \
    && apt-get update \
    && apt-get install libboost-program-options-dev libboost-filesystem-dev libboost-graph-dev libboost-system-dev libboost-test-dev libeigen3-dev libflann-dev libsuitesparse-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev libgflags-dev libsqlite3-dev libglew-dev qtbase5-dev libqt5opengl5-dev  libcgal-dev libceres-dev -y \
    && export CMAKE_BUILD_PARALLEL_LEVEL=36 \
    && export MAX_JOBS=16 \
    && export COLMAP_VERSION=dev \
    && export CUDA_ARCHITECTURES="all" \
    && git clone https://github.com/colmap/colmap.git \
    && cd colmap \
    && git reset --hard ${COLMAP_VERSION} \
    && mkdir build \
    && cd build \
    && cmake .. -GNinja -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} \
    && ninja \
    && ninja install \
    && cd ../.. \
    && rm -rf colmap \
    && apt-get clean  \
    && strip --remove-section=.note.ABI-tag /usr/lib/x86_64-linux-gnu/libQt5Core.so.5 \
    && rm -rf /var/lib/apt/lists/*
