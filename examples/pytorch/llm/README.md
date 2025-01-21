<h1 align="center">LLM SFT Example</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.1-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.0.0-6FEBB9.svg"></a>
</p>

<p align="center">
<a href="https://modelscope.cn/home">Modelscope Hub</a>
<br>
        <a href="README_CN.md">中文</a>&nbsp ｜ &nbspEnglish
</p>

## Note
1. This README.md file is **copied from** [ms-swift](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/README.md)
2. This directory has been **migrated** to [ms-swift](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm), and the files in this directory are **no longer maintained**.

## Features
1. supported sft method: [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), full(full parameter fine tuning), ...
2. supported models: [**qwen-7b**](https://github.com/QwenLM/Qwen-7B), baichuan-7b, baichuan-13b, chatglm2-6b, chatglm2-6b-32k, llama2-7b, llama2-13b, llama2-70b, openbuddy-llama2-13b, openbuddy-llama-65b, polylm-13b, ...
3. supported feature: quantization, ddp, model parallelism(device map), gradient checkpoint, gradient accumulation steps, push to modelscope hub, custom datasets, ...
4. supported datasets: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, ...

## Prepare the Environment
Experimental environment: A10, 3090, A100, ... (V100 does not support bf16, quantization)
```bash
# Installing miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh

# Setting up a conda virtual environment
conda create --name ms-sft python=3.10
conda activate ms-sft

# Setting up a global pip mirror for faster downloads
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip install torch torchvision torchaudio -U
pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib scikit-learn tqdm tensorboard -U
pip install transformers datasets -U
pip install accelerate transformers_stream_generator -U

pip install ms-swift modelscope -U
# Recommended installation from source code for faster bug fixes
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install .
# same as modelscope...(git clone ...)
```

## Run SFT and Inference
```bash
# Clone the repository and enter the code directory.
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# sft(qlora) and infer qwen-7b, Requires 16GB VRAM.
# If you want to use quantification, you need to `pip install bitsandbytes`
bash scripts/qwen_7b/qlora/sft.sh
# If you want to push the model to modelscope hub during training
bash scripts/qwen_7b/qlora/sft_push_to_hub.sh
bash scripts/qwen_7b/qlora/infer.sh

# sft(qlora+ddp) and infer qwen-7b, Requires 4*16GB VRAM.
bash scripts/qwen_7b/qlora_ddp/sft.sh
bash scripts/qwen_7b/qlora_ddp/infer.sh

# sft(full) and infer qwen-7b, Requires 95GB VRAM.
bash scripts/qwen_7b/full/sft.sh
bash scripts/qwen_7b/full/infer.sh

# For more scripts, please see `scripts/` folder
```

## Extend Datasets
1. If you need to extend the model, you can modify the `MODEL_MAPPING` in `utils/models.py`. `model_id` can be specified as a local path. In this case, `revision` doesn't work.
2. If you need to extend or customize the dataset, you can modify the `DATASET_MAPPING` in `utils/datasets.py`. You need to customize the `get_*_dataset` function, which returns a dataset with two columns: `instruction`, `output`.
