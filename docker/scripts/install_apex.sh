export MAX_JOBS=16 \
&& git clone https://github.com/NVIDIA/apex \
&& cd apex \
&& git checkout 6bd01c4b99a84648ad5e5238a959735e6936c813 \
&& TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6" pip install -v --disable-pip-version-check --no-cache --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
&& cd .. \
&& rm -rf apex
