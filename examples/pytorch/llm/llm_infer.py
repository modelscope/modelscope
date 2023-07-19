# ### Setting up experimental environment.
from _common import *


@dataclass
class Arguments:
    device: str = '0'  # e.g. '-1'; '0,1'
    seed: int = 42
    model_type: str = field(
        default='baichuan-7B',
        metadata={'choices': ['baichuan-7B', 'baichuan-13B', 'chatglm2']})
    ckpt_fpath: str = ''  # /path/to/your/iter_xxx.pth
    #
    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1
    #
    max_new_tokens: int = 512
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
        if self.lora_target_modules is None:
            if self.model_type in {'baichuan-7B', 'baichuan-13B'}:
                self.lora_target_modules = ['W_pack']
            elif self.model_type == 'chatglm2':
                self.lora_target_modules = ['query_key_value']
            else:
                raise ValueError(f'model_type: {self.model_type}')
        #
        if self.ckpt_fpath == '':
            raise ValueError('Please enter a valid fpath')


def parse_args() -> Arguments:
    args, = HfArgumentParser([Arguments]).parse_args_into_dataclasses()
    return args


args = parse_args()
select_device(args.device)

# ### Loading Model and Tokenizer
model_dir = snapshot_download('ZhipuAI/chatglm2-6b', 'v1.0.6')
model, tokenizer = get_chatglm2_model_tokenizer(model_dir)

# ### Preparing lora
lora_config = LoRAConfig(
    replace_modules=args.lora_target_modules,
    rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout_p)
logger.info(f'lora_config: {lora_config}')
Swift.prepare_model(model, lora_config)
model.bfloat16()  # Consistent with training

# ### Loading Dataset
_, test_dataset = get_alpaca_en_zh_dataset(None, True, split_seed=42)

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
        max_new_tokens=args.max_new_tokens,
        attention_mask=attention_mask,
        streamer=streamer,
        pad_token_id=tokenizer.eos_token_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True)
    print()
    print(f'[LABELS]{output}')
    print(
        '-----------------------------------------------------------------------------------'
    )
    # input('next[ENTER]')
