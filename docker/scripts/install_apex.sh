git clone https://github.com/NVIDIA/apex
cd apex
TORCH_CUDA_ARCH_LIST="6.0;6.1;6.2;7.0;7.5;8.0;8.6" pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
rm -rf apex
