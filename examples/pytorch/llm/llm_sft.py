# ### Setting up experimental environment.
"""
pip install modelscope
pip install numpy pandas matplotlib scikit-learn
pip install transformers datasets
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tqdm tensorboard torchmetrics sentencepiece charset_normalizer
pip install accelerate transformers_stream_generator

pip install numpy -U  # Resolve torchmetrics dependencies and update numpy
"""

from _common import *


@dataclass
class Arguments:
    device: str = '0,1'  # e.g. '-1'; '0'; '0,1'
    seed: int = 42
    model_type: str = field(
        default='baichuan-7b',
        metadata={
            'choices':
            ['baichuan-7b', 'baichuan-13b', 'chatglm2', 'llama2-7b']
        })
    data_sample: Optional[int] = None
    #
    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1
    #
    gradient_checkpoint: bool = True
    batch_size: int = 1
    max_epochs: int = 1
    eval_interval: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    n_accumulate_grad: int = 16
    grad_clip_norm: float = 1.
    warmup_iters: int = 200
    last_max_checkpoint_num: int = 1
    best_max_checkpoint_num: int = 1
    #
    logging_interval: int = 5
    tb_interval: int = 5

    def __post_init__(self):
        if self.lora_target_modules is None:
            if self.model_type in {'baichuan-7b', 'baichuan-13b'}:
                self.lora_target_modules = ['W_pack']
            elif self.model_type == 'chatglm2':
                self.lora_target_modules = ['query_key_value']
            elif self.model_type == 'llama2-7b':
                self.lora_target_modules = ['q_proj', 'k_proj', 'v_proj']
            else:
                raise ValueError(f'model_type: {self.model_type}')


def parse_args() -> Arguments:
    args, = HfArgumentParser([Arguments]).parse_args_into_dataclasses()
    return args


args = parse_args()
logger.info(args)
select_device(args.device)
seed_everything(args.seed)

# ### Loading Model and Tokenizer
if args.model_type == 'baichuan-7b':
    model_dir = snapshot_download('baichuan-inc/baichuan-7B', 'v1.0.5')
    model, tokenizer = get_baichuan_model_tokenizer(model_dir)
elif args.model_type == 'baichuan-13b':
    model_dir = snapshot_download('baichuan-inc/Baichuan-13B-Base', 'v1.0.2')
    model, tokenizer = get_baichuan_model_tokenizer(model_dir)
elif args.model_type == 'chatglm2':
    model_dir = snapshot_download('ZhipuAI/chatglm2-6b', 'v1.0.6')
    model, tokenizer = get_chatglm2_model_tokenizer(model_dir)
elif args.model_type == 'llama2-7b':
    model_dir = snapshot_download('modelscope/Llama-2-7b-ms', 'v1.0.0')
    model, tokenizer = get_llama2_model_tokenizer(model_dir)
else:
    raise ValueError(f'model_type: {args.model_type}')

#
if args.gradient_checkpoint:
    # baichuan13B does not implement the `get_input_embeddings` function
    if args.model_type == 'baichuan-13b':

        def get_input_embeddings(self):
            return self.model.embed_tokens

        model.__class__.get_input_embeddings = get_input_embeddings.__get__(
            model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

# ### Preparing lora
lora_config = LoRAConfig(
    replace_modules=args.lora_target_modules,
    rank=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout_p)
logger.info(f'lora_config: {lora_config}')
Swift.prepare_model(model, lora_config)
#
show_freeze_layers(model)
print_model_info(model)
_p: Parameter = list(model.parameters())[100]
logger.info(f'device: {_p.device}, dtype: {_p.dtype}')
model.bfloat16()

# ### Loading Dataset
tokenize_function = partial(tokenize_function, tokenizer=tokenizer)
train_dataset, val_dataset = get_alpaca_en_zh_dataset(
    tokenize_function, split_seed=42, data_sample=args.data_sample)
# Data analysis
stat_dataset(train_dataset)
stat_dataset(val_dataset)
data_collate_fn = partial(data_collate_fn, tokenizer=tokenizer)
print_example(train_dataset[0], tokenizer)

# ### Setting Config
cfg_file = os.path.join(model_dir, 'configuration.json')
#
T_max = get_T_max(len(train_dataset), args.batch_size, args.max_epochs, True)
work_dir = get_work_dir(f'runs/{args.model_type}')
config = Config({
    'train': {
        'dataloader': {
            'batch_size_per_gpu': args.batch_size,
            'workers_per_gpu': 1,
            'shuffle': True,
            'drop_last': True,
            'pin_memory': True
        },
        'max_epochs':
        args.max_epochs,
        'work_dir':
        work_dir,
        'optimizer': {
            'type': 'AdamW',
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
            'options': {
                'cumulative_iters': args.n_accumulate_grad,
                'grad_clip': {
                    'norm_type': 2,
                    'max_norm': args.grad_clip_norm
                }
            }
        },
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': T_max,
            'eta_min': 0,
            'options': {
                'by_epoch': False,
                'warmup': {
                    'type': 'LinearWarmup',
                    'warmup_ratio': 0.1,
                    'warmup_iters': args.warmup_iters
                }
            }
        },
        'hooks': [
            {
                'type': 'CheckpointHook',
                'by_epoch': False,
                'interval': args.eval_interval,
                'max_checkpoint_num': args.last_max_checkpoint_num
            },
            {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': args.eval_interval
            },
            {
                'type': 'BestCkptSaverHook',
                'metric_key': 'loss',
                'save_best': True,
                'rule': 'min',
                'max_checkpoint_num': args.best_max_checkpoint_num
            },
            {
                'type': 'TextLoggerHook',
                'by_epoch': True,  # Whether EpochBasedTrainer is used
                'interval': args.logging_interval
            },
            {
                'type': 'TensorboardHook',
                'by_epoch': False,
                'interval': args.tb_interval
            }
        ]
    },
    'evaluation': {
        'dataloader': {
            'batch_size_per_gpu': args.batch_size,
            'workers_per_gpu': 1,
            'shuffle': False,
            'drop_last': False,
            'pin_memory': True
        },
        'metrics': [{
            'type': 'my_metric',
            'vocab_size': tokenizer.vocab_size
        }]
    }
})

# ### Finetuning


def cfg_modify_fn(cfg: Config) -> Config:
    cfg.update(config)
    return cfg


trainer = EpochBasedTrainer(
    model=model,
    cfg_file=cfg_file,
    data_collator=data_collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    remove_unused_data=True,
    seed=42,
    cfg_modify_fn=cfg_modify_fn,
)

trainer.train()

# ### Visualization
tb_dir = os.path.join(work_dir, 'tensorboard_output')
plot_image(tb_dir, ['loss'], 0.9)
