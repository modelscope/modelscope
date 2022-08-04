# Trainer使用教程
Modelscope提供了众多预训练模型，你可以使用其中任意一个，利用公开数据集或者私有数据集针对特定任务进行模型训练，在本篇文章中将介绍如何使用Modelscope的`Trainer`模块进行Finetuning和评估。

## 环境准备
详细步骤可以参考 [快速开始](../quick_start.md)

### 准备数据集

在开始Finetuning前，需要准备一个数据集用以训练和评估，详细可以参考数据集使用教程。

```python
from datasets import Dataset
train_dataset = MsDataset.load'afqmc_small', namespace='modelscope', split='train')
eval_dataset = MsDataset.load('afqmc_small', namespace='modelscope', split='validation')
```
### 训练
ModelScope把所有训练相关的配置信息全部放到了模型仓库下的`configuration.json`中，因此我们只需要创建Trainer，加载配置文件，传入数据集即可完成训练。

首先，通过工厂方法创建Trainer， 需要传入模型仓库路径， 训练数据集对象，评估数据集对象，训练目录
```python
kwargs = dict(
    model='damo/nlp_structbert_sentiment-classification_chinese-base',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir='work_dir')

trainer = build_trainer(default_args=kwargs)
```

启动训练。
```python
trainer.train()
```

如果需要调整训练参数，可以在模型仓库页面下载`configuration.json`文件到本地，修改参数后，指定配置文件路径，创建trainer
```python
kwargs = dict(
    model='damo/nlp_structbert_sentiment-classification_chinese-base',
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    cfg_file='你的配置文件路径'
    work_dir='work_dir')

trainer = build_trainer(default_args=kwargs)
trainer.train()
```


### 评估
训练过程中会定期使用验证集进行评估测试， Trainer模块也支持指定特定轮次保存的checkpoint路径，进行单次评估。
```python
eval_results = trainer.evaluate('work_dir/epoch_10.pth')
print(eval_results)
```
