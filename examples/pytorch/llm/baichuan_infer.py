# ### Setting up experimental environment.
from _common import *
from transformers import TextStreamer

device_ids = [0, 1]
select_device(device_ids)

# ### Loading Model and Tokenizer
# Note: You need to set the value of `CKPT_FPATH`
BAICHUAN_TYPE = '13B'  # Literal['7B', '13B']
CKPT_FAPTH = '/path/to/your/iter_xxx.pth'
LORA_TARGET_MODULES = ['W_pack']

if BAICHUAN_TYPE == '7B':
    model_dir = snapshot_download('baichuan-inc/baichuan-7B', 'v1.0.5')
    model, tokenizer = get_baichuan7B_model_tokenizer(model_dir)
else:
    model_dir = snapshot_download('baichuan-inc/Baichuan-13B-Base', 'v1.0.2')
    model, tokenizer = get_baichuan13B_model_tokenizer(model_dir)
model.bfloat16()  # Consistent with training

# ### Preparing lora
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT_P = 0  # Arbitrary value
lora_config = LoRAConfig(
    replace_modules=LORA_TARGET_MODULES,
    rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT_P,
    pretrained_weights=CKPT_FAPTH)
logger.info(f'lora_config: {lora_config}')
Swift.prepare_model(model, lora_config)

# ### Loading Dataset
_, test_dataset = get_alpaca_en_zh_dataset(None, True)

# ### Inference
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
mini_test_dataset = test_dataset.select(range(5))
for d in mini_test_dataset:
    output = d['output']
    d['output'] = None
    input_ids = tokenize_function(d, tokenizer)['input_ids']
    print(f'[TEST]{tokenizer.decode(input_ids)}', end='')
    input_ids = torch.tensor(input_ids)[None].cuda()
    attention_mask = torch.ones_like(input_ids)
    generate_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=512,
        attention_mask=attention_mask,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.7,
        do_sample=True)
    print()
    print(f'[LABELS]{output}')
    print(
        '-----------------------------------------------------------------------------------'
    )
    # input('next[ENTER]')
