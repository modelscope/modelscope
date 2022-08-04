# 快速开始
ModelScope Library目前支持tensorflow，pytorch深度学习框架进行模型训练、推理， 在Python 3.7+, Pytorch 1.8+, Tensorflow1.15，Tensorflow 2.x上测试可运行。

注： `语音相关`的功能仅支持 python3.7,tensorflow1.15的`linux`环境使用。  其他功能可以在windows、mac上安装使用。

## python环境配置
首先，参考[文档](https://docs.anaconda.com/anaconda/install/) 安装配置Anaconda环境

安装完成后，执行如下命令为modelscope library创建对应的python环境。
```shell
conda create -n modelscope python=3.7
conda activate modelscope
```
## 安装深度学习框架
* 安装pytorch[参考链接](https://pytorch.org/get-started/locally/)
```shell
pip install torch torchvision
```
* 安装Tensorflow[参考链接](https://www.tensorflow.org/install/pip)
```shell
pip install --upgrade tensorflow
```
## ModelScope library 安装

注： 如果在安装过程中遇到错误，请前往[常见问题](faq.md)查找解决方案。

### pip安装
执行如下命令：
```shell
pip install "modelscope[all]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```

如需体验`语音功能`，请`额外`执行如下命令：
```shell
pip install "modelscope[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
```
### 使用源码安装
适合本地开发调试使用，修改源码后可以直接执行
下载源码可以直接clone代码到本地
```shell
git clone git@gitlab.alibaba-inc.com:Ali-MaaS/MaaS-lib.git modelscope
git fetch origin master
git checkout master
cd modelscope
```
安装依赖并设置PYTHONPATH
```shell
pip install -e ".[all]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html
export PYTHONPATH=`pwd`
```
注： 6.30版本需要把cv、nlp、multi-modal领域依赖都装上，7.30号各个领域依赖会作为选装，用户需要使用哪个领域安装对应领域依赖即可。

如需使用语音功能，请执行如下命令安装语音功能所需依赖
```shell
pip install -e ".[audio]" -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo
```

### 安装验证
安装成功后，可以执行如下命令进行验证安装是否正确
```shell
python -c "from modelscope.pipelines import pipeline;print(pipeline('word-segmentation')('今天天气不错，适合 出去游玩'))"
{'output': '今天 天气 不错 ， 适合 出去 游玩'}
```
## 推理

pipeline函数提供了简洁的推理接口，相关介绍和示例请参考[pipeline使用教程](tutorials/pipeline.md)

## 训练 & 评估

Trainer类提供了简洁的Finetuning和评估接口，相关介绍和示例请参考[Trainer使用教程](tutorials/trainer.md)
