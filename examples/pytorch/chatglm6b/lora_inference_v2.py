from modelscope import Model, pipeline, read_config
from modelscope.metainfo import Models
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from modelscope.utils.config import ConfigDict

lora_config = LoRAConfig(
    replace_modules=['attention.query_key_value'],
    rank=32,
    lora_alpha=32,
    lora_dropout=0.05,
    pretrained_weights='./lora_dureader_target/iter_600.pth')

model_dir = 'ZhipuAI/chatglm2-6b'
model_config = read_config(model_dir)
model_config['model'] = ConfigDict({
    'type': Models.chatglm2_6b,
})

model = Model.from_pretrained(model_dir, cfg_dict=model_config)
model = model.bfloat16()
Swift.prepare_model(model, lora_config)

pipe = pipeline('chat', model, pipeline_name='chatglm2_6b-text-generation')

print(
    pipe({
        'text':
        '纵使进入21世纪后，我国教育水平有了明显进步，高考的难度却依旧不容小觑，高考被中国学生和家长定义为改变命运、改写人生脑重要考试，为了这场考试，学生和家长都付出了很多。',
        'history': []
    }))
