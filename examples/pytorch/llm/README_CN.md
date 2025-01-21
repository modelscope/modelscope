<h1 align="center">大模型微调的例子</h1>

<p align="center">
<img src="https://img.shields.io/badge/python-%E2%89%A53.8-5be.svg">
<img src="https://img.shields.io/badge/pytorch-%E2%89%A51.12%20%7C%20%E2%89%A52.0-orange.svg">
<a href="https://github.com/modelscope/modelscope/"><img src="https://img.shields.io/badge/modelscope-%E2%89%A51.8.1-5D91D4.svg"></a>
<a href="https://github.com/modelscope/swift/"><img src="https://img.shields.io/badge/ms--swift-%E2%89%A51.0.0-6FEBB9.svg">
</p>

<p align="center">
<a href="https://modelscope.cn/home">魔搭社区</a>
<br>
        中文&nbsp ｜ &nbsp<a href="README.md">English</a>
</p>

## 请注意
1. 该README_CN.md**拷贝**自[ms-swift](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm/README_CN.md)
2. 该目录已经**迁移**至[ms-swift](https://github.com/modelscope/swift/tree/main/examples/pytorch/llm), 此目录中的文件**不再维护**.

## 特性
1. [lora](https://arxiv.org/abs/2106.09685), [qlora](https://arxiv.org/abs/2305.14314), 全参数微调, ...
2. 支持的模型: [**qwen-7b**](https://github.com/QwenLM/Qwen-7B), baichuan-7b, baichuan-13b, chatglm2-6b, chatglm2-6b-32k, llama2-7b, llama2-13b, llama2-70b, openbuddy-llama2-13b, openbuddy-llama-65b, polylm-13b, ...
3. 支持的特性: 模型量化, DDP, 模型并行(device_map), gradient checkpoint, 梯度累加, 支持推送modelscope hub, 支持自定义数据集, ...
4. 支持的数据集: alpaca-en(gpt4), alpaca-zh(gpt4), finance-en, multi-alpaca-all, code-en, instinwild-en, instinwild-zh, ...


## 准备实验环境
实验环境: A10, 3090, A100均可. (V100不支持bf16, 量化)
```bash
# 安装miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 一直[ENTER], 最后一个选项yes即可
sh Miniconda3-latest-Linux-x86_64.sh

# conda虚拟环境搭建
conda create --name ms-sft python=3.10
conda activate ms-sft

# pip设置全局镜像与相关python包安装
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

pip install torch torchvision torchaudio -U
pip install sentencepiece charset_normalizer cpm_kernels tiktoken -U
pip install matplotlib scikit-learn tqdm tensorboard -U
pip install transformers datasets -U
pip install accelerate transformers_stream_generator -U

pip install ms-swift modelscope -U
# 推荐从源码安装swift和modelscope, 这具有更多的特性和更快的bug修复
git clone https://github.com/modelscope/swift.git
cd swift
pip install -r requirements.txt
pip install .
# modelscope类似...(git clone ...)
```

## 微调和推理
```bash
# clone仓库并进入代码目录
git clone https://github.com/modelscope/swift.git
cd swift/examples/pytorch/llm

# 微调(qlora)+推理 qwen-7b, 需要16GB显存.
# 如果你想要使用量化, 你需要`pip install bitsandbytes`
bash scripts/qwen_7b/qlora/sft.sh
# 如果你想在训练时, 将权重push到modelscope hub中.
bash scripts/qwen_7b/qlora/sft_push_to_hub.sh
bash scripts/qwen_7b/qlora/infer.sh

# 微调(qlora+ddp)+推理 qwen-7b, 需要4卡*16GB显存.
bash scripts/qwen_7b/qlora_ddp/sft.sh
bash scripts/qwen_7b/qlora_ddp/infer.sh

# 微调(full)+推理 qwen-7b, 需要95G显存.
bash scripts/qwen_7b/full/sft.sh
bash scripts/qwen_7b/full/infer.sh

# 更多的scripts脚本, 可以看`scripts`文件夹
```

## 拓展数据集
1. 如果你想要拓展模型, 你可以修改`utils/models.py`文件中的`MODEL_MAPPING`. `model_id`可以指定为本地路径, 这种情况下, `revision`参数不起作用.
2. 如果你想要拓展或使用自定义数据集, 你可以修改`utils/datasets.py`文件中的`DATASET_MAPPING`. 你需要自定义`get_*_dataset`函数, 并返回包含`instruction`, `output`两列的数据集.
