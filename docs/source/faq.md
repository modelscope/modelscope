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
3. 安装tokenizers
    ```shell
    pip install tokenizers
    ```
reference: [https://huggingface.co/docs/tokenizers/installation#installation-from-sources](https://huggingface.co/docs/tokenizers/installation#installation-from-sources)

### 2. pip 安装包冲突

> ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.

由于依赖库之间的版本不兼容，可能会存在版本冲突的情况，大部分情况下不影响正常运行。

### 3. 安装pytorch出现版本错误

> ERROR: Ignored the following versions that require a different python version: 1.1.0 Requires-Python >=3.8; 1.1.0rc1 Requires-Python >=3.8; 1.1.1 Requires-Python >=3.8
> ERROR: Could not find a version that satisfies the requirement torch==1.8.1+cu111 (from versions: 1.0.0, 1.0.1, 1.0.1.post2, 1.1.0, 1.2.0, 1.3.0, 1.3.1, 1.4.0, 1.5.0, 1.5.1, 1.6.0, 1.7.0, 1.7.1, 1.8.0, 1.8.1, 1.9.0, 1.9.1, 1.10.0, 1.10.1, 1.10.2, 1.11.0)
> ERROR: No matching distribution found for torch==1.8.1+cu111

安装时使用如下命令：

```shell
pip install -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```
### 4. zsh: no matches found: modelscope-0.2.2-py3-none-any.whl[all]
mac终端的zsh 对于[]需要做转义，执行如下命令
```shell
pip install modelscope\[all\] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
