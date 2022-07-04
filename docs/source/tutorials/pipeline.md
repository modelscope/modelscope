# Pipeline使用教程
本文简单介绍如何使用`pipeline`函数加载模型进行推理。`pipeline`函数支持按照任务类型、模型名称从模型仓库拉取模型进行进行推理，包含以下几个方面：
* 使用pipeline()函数进行推理
* 指定特定预处理、特定模型进行推理
* 不同场景推理任务示例
## 环境准备
详细步骤可以参考 [快速开始](../quick_start.md)
## Pipeline基本用法
下面以中文分词任务为例，说明pipeline函数的基本用法

1. pipeline函数支持指定特定任务名称，加载任务默认模型，创建对应pipeline对象
   执行如下python代码
   ```python
   from modelscope.pipelines import pipeline
   word_segmentation = pipeline('word-segmentation')
   ```

2. 输入文本
   ``` python
   input = '今天天气不错，适合出去游玩'
   print(word_segmentation(input))
   {'output': '今天 天气 不错 ， 适合 出去 游玩'}
   ```

3. 输入多条样本

pipeline对象也支持传入多个样本列表输入，返回对应输出列表，每个元素对应输入样本的返回结果

   ```python
   inputs =  ['今天天气不错，适合出去游玩','这本书很好，建议你看看']
   print(word_segmentation(inputs))
   [{'output': '今天 天气 不错 ， 适合 出去 游玩'}, {'output': '这 本 书 很 好 ， 建议 你 看看'}]
   ```
## 指定预处理、模型进行推理
pipeline函数支持传入实例化的预处理对象、模型对象，从而支持用户在推理过程中定制化预处理、模型。

1. 首先，创建预处理方法和模型
```python
from modelscope.models import Model
from modelscope.preprocessors import TokenClassifcationPreprocessor
model = Model.from_pretrained('damo/nlp_structbert_word-segmentation_chinese-base')
tokenizer = TokenClassifcationPreprocessor(model.model_dir)
```

2. 使用tokenizer和模型对象创建pipeline
```python
from modelscope.pipelines import pipeline
word_seg = pipeline('word-segmentation', model=model, preprocessor=tokenizer)
input = '今天天气不错，适合出去游玩'
print(word_seg(input))
{'output': '今天 天气 不错 ， 适合 出去 游玩'}
```
## 不同场景任务推理示例
下面以一个图像任务：人像抠图（'image-matting'）为例，进一步说明pipeline的用法
```python
import cv2
from modelscope.pipelines import pipeline
img_matting = pipeline('image-matting')
result = img_matting('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/image_matting.png')
cv2.imwrite('result.png', result['output_png'])
```
