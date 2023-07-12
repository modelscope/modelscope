# ### Setting up experimental environment.
from _common import *
from transformers import TextStreamer

device_ids = [0, 1]
logger.info(device_ids)
select_device(device_ids)

# ### Loading Model and Tokenizer
# Note: You need to set the value of `CKPT_FPATH`
CKPT_FAPTH = '/path/to/your/xxx.pth'
LORA_TARGET_MODULES = ['query_key_value']

model, tokenizer = get_chatglm2_model_tokenizer()
if tokenizer.eos_token_id is None:
    tokenizer.eos_token_id = tokenizer.pad_token_id
if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = 1
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
for d in test_dataset[:5]:
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
        pad_token_id=tokenizer.pad_token_id,
        temperature=0.7,
        top_k=50,
        do_sample=True)
    print()
    print(f'[LABELS]{output}')
    print(
        '-----------------------------------------------------------------------------------'
    )
    # input('next[ENTER]')
