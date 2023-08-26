import os.path as osp

import torch
from swift import LoRAConfig, Swift

from modelscope import Model, pipeline, read_config
from modelscope.metainfo import Models
from modelscope.utils.config import ConfigDict

lora_config = LoRAConfig(
    target_modules=['attention.query_key_value'],
    r=32,
    lora_alpha=32,
    lora_dropout=0.05)

model_dir = 'ZhipuAI/ChatGLM-6B'
model_config = read_config(model_dir)
model_config['model'] = ConfigDict({
    'type': Models.chatglm_6b,
})

model = Model.from_pretrained(model_dir, cfg_dict=model_config)
model = model.bfloat16()
model = Swift.prepare_model(model, lora_config)
work_dir = './tmp'
state_dict = torch.load(osp.join(work_dir, 'iter_600.pth'))
model = Swift.from_pretrained(
    model, osp.join(work_dir, 'output_best'), device_map='auto')
model.load_state_dict(state_dict)
pipe = pipeline('chat', model, pipeline_name='chatglm6b-text-generation')

print(
    pipe({
        'text':
        '纵使进入21世纪后，我国教育水平有了明显进步，高考的难度却依旧不容小觑，高考被中国学生和家长定义为改变命运、改写人生脑重要考试，为了这场考试，学生和家长都付出了很多。',
        'history': []
    }))
