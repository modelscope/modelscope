# Pipeline使用教程

本文将简单介绍如何使用`pipeline`函数加载模型进行推理。`pipeline`函数支持按照任务类型、模型名称从模型仓库
拉取模型进行进行推理，当前支持的任务有

* 人像抠图 (image-matting)
* 基于bert的语义情感分析 (bert-sentiment-analysis)

本文将从如下方面进行讲解如何使用Pipeline模块：
* 使用pipeline()函数进行推理
* 指定特定预处理、特定模型进行推理
* 不同场景推理任务示例

## 环境准备
详细步骤可以参考 [快速开始](../quick_start.md)

## Pipeline基本用法

1. pipeline函数支持指定特定任务名称，加载任务默认模型，创建对应Pipeline对象
   注： 当前还未与modelhub进行打通，需要手动下载模型，创建pipeline时需要指定本地模型路径，未来会支持指定模型名称从远端仓库
   拉取模型并初始化。

   下载模型文件
   ```shell
   wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/matting_person.pb
   ```
   执行如下python代码
   ```python
   >>> from maas_lib.pipelines import pipeline
   >>> img_matting = pipeline(task='image-matting', model='damo/image-matting-person')
   ```

2. 传入单张图像url进行处理
   ``` python
   >>> import cv2
   >>> result = img_matting('http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png')
   >>> cv2.imwrite('result.png', result['output_png'])
   >>> import os.path as osp
   >>> print(f'result file path is {osp.abspath("result.png")}')
   ```

   pipeline对象也支持传入一个列表输入，返回对应输出列表，每个元素对应输入样本的返回结果
   ```python
   >>> results = img_matting(
       [
           'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png',
           'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png',
           'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png',
       ])
   ```

   如果pipeline对应有一些后处理参数，也支持通过调用时候传入.
   ```python
   >>> pipe = pipeline(task_name)
   >>> result = pipe(input, post_process_args)
   ```

## 指定预处理、模型进行推理
pipeline函数支持传入实例化的预处理对象、模型对象，从而支持用户在推理过程中定制化预处理、模型。
下面以文本情感分类为例进行介绍。

由于demo模型为EasyNLP提供的模型，首先，安装EasyNLP
```shell
pip install https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/package/whl/easynlp-0.0.4-py2.py3-none-any.whl
```


下载模型文件
```shell
wget https://atp-modelzoo-sh.oss-cn-shanghai.aliyuncs.com/release/easynlp_modelzoo/alibaba-pai/bert-base-sst2.zip && unzip bert-base-sst2.zip
```

创建tokenizer和模型
```python
>>> from maas_lib.models import Model
>>> from maas_lib.preprocessors import SequenceClassificationPreprocessor
>>> model = Model.from_pretrained('damo/bert-base-sst2')
>>> tokenizer = SequenceClassificationPreprocessor(
            model.model_dir, first_sequence='sentence', second_sequence=None)
```

使用tokenizer和模型对象创建pipeline
```python
>>> from maas_lib.pipelines import pipeline
>>> semantic_cls = pipeline('text-classification', model=model,   preprocessor=tokenizer)
>>> semantic_cls("Hello world!")
```

## 不同场景任务推理示例

人像抠图、语义分类建上述两个例子。  其他例子未来添加。
