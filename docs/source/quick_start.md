# 快速开始

## python环境配置
首先，参考[文档](https://docs.anaconda.com/anaconda/install/) 安装配置Anaconda环境

安装完成后，执行如下命令为maas library创建对应的python环境。
```shell
conda create -n maas python=3.6
conda activate maas
```
检查python和pip命令是否切换到conda环境下。
```shell
which python
# ~/workspace/anaconda3/envs/maas/bin/python

which pip
# ~/workspace/anaconda3/envs/maas/bin/pip
```
注： 本项目只支持`python3`环境，请勿使用python2环境。

## 第三方依赖安装

MaaS Library目前支持tensorflow，pytorch两大深度学习框架进行模型训练、推理， 在Python 3.6+,  Pytorch 1.8+, Tensorflow 2.6上测试可运行，用户可以根据所选模型对应的计算框架进行安装，可以参考如下链接进行安装所需框架:

* [Pytorch安装指导](https://pytorch.org/get-started/locally/)
* [Tensorflow安装指导](https://www.tensorflow.org/install/pip)


## MaaS library 安装

注： 如果在安装过程中遇到错误，请前往[常见问题](faq.md)查找解决方案。

### pip安装
```shell
pip install -r http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/maas/maas.txt
```

安装成功后，可以执行如下命令进行验证安装是否正确
```shell
python -c "from maas_lib.pipelines import pipeline;print(pipeline('image-matting',model='damo/image-matting-person')('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'))"
```


### 使用源码安装

适合本地开发调试使用，修改源码后可以直接执行
```shell
git clone git@gitlab.alibaba-inc.com:Ali-MaaS/MaaS-lib.git maaslib
git fetch origin master
git checkout master

cd maaslib

#安装依赖
pip install -r requirements.txt

# 设置PYTHONPATH
export PYTHONPATH=`pwd`
```

安装成功后，可以执行如下命令进行验证安装是否正确
```shell
python -c "from maas_lib.pipelines import pipeline;print(pipeline('image-matting',model='damo/image-matting-person')('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'))"
```


## 训练

to be done

## 评估

to be done

## 推理

pipeline函数提供了简洁的推理接口，示例如下， 更多pipeline介绍和示例请参考[pipeline使用教程](tutorials/pipeline.md)

```python
import cv2
import os.path as osp
from maas_lib.pipelines import pipeline
from maas_lib.utils.constant import Tasks

# 根据任务名创建pipeline
img_matting = pipeline(Tasks.image_matting, model='damo/image-matting-person')

# 直接提供图像文件的url作为pipeline推理的输入
result = img_matting(
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
)
cv2.imwrite('result.png', result['output_png'])
print(f'Output written to {osp.abspath("result.png")}')

```

此外，pipeline接口也能接收Dataset作为输入，上面的代码同样可以实现为

```python
import cv2
import os.path as osp
from maas_lib.pipelines import pipeline
from maas_lib.utils.constant import Tasks
from maas_lib.pydatasets import PyDataset

# 使用图像url构建PyDataset，此处也可通过 input_location = '/dir/to/images' 来使用本地文件夹
input_location = [
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
]
dataset = PyDataset.load(input_location, target='image')
img_matting = pipeline(Tasks.image_matting, model='damo/image-matting-person')
# 输入为PyDataset时，输出的结果为迭代器
result = img_matting(dataset)
cv2.imwrite('result.png', next(result)['output_png'])
print(f'Output written to {osp.abspath("result.png")}')
```
