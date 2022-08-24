ModelScope Library目前支持tensorflow，pytorch深度学习框架进行模型训练、推理， 在Python 3.7+, Pytorch 1.8+, Tensorflow1.15，Tensorflow 2.x上测试可运行。

**注： **`**语音相关**`**的功能仅支持 python3.7,tensorflow1.15的**`**linux**`**环境使用。  其他功能可以在windows、mac上安装使用。**

## python环境配置

首先，参考[文档](https://docs.anaconda.com/anaconda/install/) 安装配置Anaconda环境。
安装完成后，执行如下命令为modelscope library创建对应的python环境。

```shell
conda create -n modelscope python=3.7
conda activate modelscope
```

## 安装深度学习框架

- 安装pytorch[参考链接](https://pytorch.org/get-started/locally/)。

```shell
pip3 install torch torchvision torchaudio
```

- 安装Tensorflow[参考链接](https://www.tensorflow.org/install/pip)。

```shell
pip install --upgrade tensorflow
```

## ModelScope library 安装

注： 如果在安装过程中遇到错误，请前往[常见问题](faq.md)查找解决方案。

### pip安装
执行如下命令可以安装所有领域依赖：
```shell
pip install "modelscope[cv,nlp,audio,multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验`语音功能`，请执行如下命令：
```shell
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验CV功能，可执行如下命令安装依赖：
```shell
pip install "modelscope[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验NLP功能，可执行如下命令安装依赖：
```shell
pip install "modelscope[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验多模态功能，可执行如下命令安装依赖：
```shell
pip install "modelscope[multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
**注**：

1. `**语音相关**`**的功能仅支持 python3.7,tensorflow1.15的**`**linux**`**环境使用。  其他功能可以在windows、mac上安装使用。**

2. 语音领域中一部分模型使用了三方库SoundFile进行wav文件处理，**在Linux系统上用户需要手动安装SoundFile的底层依赖库libsndfile**，在Windows和MacOS上会自动安装不需要用户操作。详细信息可参考[SoundFile官网](https://github.com/bastibe/python-soundfile#installation)。以Ubuntu系统为>例，用户需要执行如下命令:

    ```shell
    sudo apt-get update
    sudo apt-get install libsndfile1
    ```

3. **CV功能使用需要安装mmcv-full， 请参考mmcv**[**安装手册**](https://github.com/open-mmlab/mmcv#installation)**进行安装**

### 使用源码安装

适合本地开发调试使用，修改源码后可以直接执行。
ModelScope的源码可以直接clone到本地：

```shell
git clone git@gitlab.alibaba-inc.com:Ali-MaaS/MaaS-lib.git modelscope
cd modelscope
git fetch origin master
git checkout master

```


安装依赖
如需安装所有依赖，请执行如下命令
```shell
pip install -e ".[cv,nlp,audio,multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```



如需体验`语音功能`，请单独执行如下命令：
```shell
pip install -e ".[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验CV功能，可执行如下命令安装依赖：
```shell
pip install -e ".[cv]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
如仅需体验NLP功能，可执行如下命令安装依赖：
```shell
pip install -e ".[nlp]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如仅需体验多模态功能，可执行如下命令安装依赖：
```shell
pip install -e ".[multi-modal]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

### 安装验证

安装成功后，可以执行如下命令进行验证安装是否正确：

```shell
python -c "from modelscope.pipelines import pipeline;print(pipeline('word-segmentation')('今天天气不错，适合 出去游玩'))"
```
