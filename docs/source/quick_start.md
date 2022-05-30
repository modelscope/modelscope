# 快速开始

## 环境准备

方式一： whl包安装， 执行如下命令
```shell
pip install http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/release/maas_lib-0.1.0-py3-none-any.whl
```

方式二： 源码环境指定， 适合本地开发调试使用，修改源码后可以直接执行
```shell
git clone git@gitlab.alibaba-inc.com:Ali-MaaS/MaaS-lib.git maaslib
git fetch origin release/0.1
git checkout release/0.1

cd maaslib

#安装依赖
pip install -r requirements.txt

# 设置PYTHONPATH
export PYTHONPATH=`pwd`
```

备注： mac arm cpu暂时由于依赖包版本问题会导致requirements暂时无法安装，请使用mac intel cpu， linux cpu/gpu机器测试。


## 训练

to be done

## 评估

to be done

## 推理
to be done
<!-- pipeline函数提供了简洁的推理接口，示例如下

注： 这里提供的接口是完成和modelhub打通后的接口，暂时不支持使用。pipeline使用示例请参考 [pipelien tutorial](tutorials/pipeline.md)给出的示例。

```python
import cv2
from maas_lib.pipelines import pipeline

# 根据任务名创建pipeline
img_matting = pipeline('image-matting')

# 根据任务和模型名创建pipeline
img_matting = pipeline('image-matting', model='damo/image-matting-person')

# 自定义模型和预处理创建pipeline
model = Model.from_pretrained('damo/xxx')
preprocessor = Preprocessor.from_pretrained(cfg)
img_matting = pipeline('image-matting', model=model, preprocessor=preprocessor)

# 推理
result = img_matting(
                'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/test/maas/image_matting/test.png'
            )

# 保存结果图片
cv2.imwrite('result.png', result['output_png'])
``` -->
