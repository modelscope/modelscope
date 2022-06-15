# 常见问题

<a name="macos-pip-tokenizer-error"></a>

### 1. macOS环境pip方式安装tokenizers报错

对于tokenizers库， pypi上缺乏针对`macOS`环境预编译包，需要搭建源码编译环境后才能正确安装，步骤如下：

1. 安装rust
    ```shell
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    pip install setuptools_rust

    ```

2. 更新rust环境变量

    ```shell
    source $HOME/.cargo/env
    ```
3. 安装tokenziers
    ```shell
    pip install tokenziers
    ```
reference: [https://huggingface.co/docs/tokenizers/installation#installation-from-sources](https://huggingface.co/docs/tokenizers/installation#installation-from-sources)

### 2. pip 安装包冲突

> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

由于依赖库之间的版本不兼容，可能会存在版本冲突的情况，大部分情况下不影响正常运行。
