import os.path as osp
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.swift.lora import LoRAConfig
from modelscope.swift import Swift

# 使用源模型 model_id 初始化 pipeline
model_id = 'baichuan-inc/baichuan-7B'
pipe = pipeline(task=Tasks.text_generation, model=model_id, model_revision='v1.0.2')
# lora 配置，replace_modules，rank，alpha 需与训练参数相同
lora_config = LoRAConfig(
    replace_modules=['pack'],
    rank=32,
    lora_alpha=32)
# 转 bf16，需与训练精度相同
model = pipe.model.bfloat16()
# model 转 lora
Swift.prepare_model(model, lora_config)
# 加载 lora 参数，默认 link 到于 output/model 路径
work_dir = './tmp'
state_dict = torch.load(osp.join(work_dir, 'output/pytorch_model.bin'))
model.load_state_dict(state_dict)
# 使用 lora model 替换 pipeline 中的 model
pipe.model = model
# 使用 pipeline 推理
result_zh = pipe('今天天气是真的')
print(result_zh)

