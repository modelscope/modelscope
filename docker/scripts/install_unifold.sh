apt-get update && apt-get install -y  hmmer kalign curl cmake \
        && apt-get clean && rm -rf /var/lib/apt/lists/* \
        && git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
        && mkdir /tmp/hh-suite/build \
        && pushd /tmp/hh-suite/build \
        && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
        && make -j 4 && make install \
        && ln -s /opt/hhsuite/bin/* /usr/bin \
        && popd \
        && rm -rf /tmp/hh-suite \
        && pip install --no-cache-dir unicore -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html \
        && pip install --no-cache-dir  biopython ipdb
