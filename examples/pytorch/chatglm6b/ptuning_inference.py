import torch

from modelscope import Model, pipeline, read_config
from modelscope.metainfo import Models
from modelscope.utils.config import ConfigDict

model_dir = 'ZhipuAI/ChatGLM-6B'
model_config = read_config(model_dir)
model_config['model'] = ConfigDict({
    'type': Models.chatglm_6b,
    'pre_seq_len': 128,
    'prefix_projection': False,
})

model = Model.from_pretrained(model_dir, cfg_dict=model_config)
model = model.half()
model.transformer.prefix_encoder.float()
prefix_state_dict = torch.load('./ptuning_dureader_target/iter_900.pth')
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith('transformer.prefix_encoder.'):
        new_prefix_state_dict[k[len('transformer.prefix_encoder.'):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

pipe = pipeline('chat', model)

print(
    pipe({
        'text':
        '维生素C也叫抗坏血酸，所以它最重要的一个作用是预防坏血病。另外，维生素C在控制感染和愈合伤口方面发挥作用，是一种强大的抗氧化剂，'
        '可以中和有害的自由基。维生素C还是合成胶原蛋白的重要营养成分，胶原蛋白是结缔组织中的一种纤维蛋白，它存在于身体的各个系统中：'
        '神经系统、免疫系统、骨骼系统、软骨系统、血液系统和其他系统。维生素C有助于产生作用于大脑和神经的多种激素和化学信使。',
        'history': []
    }))
