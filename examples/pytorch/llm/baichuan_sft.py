# ### Setting up experimental environment.
"""
pip install modelscope
pip install numpy pandas matplotlib scikit-learn
pip install transformers datasets
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install tensorboard
pip install torchmetrics
pip install sentencepiece
pip install accelerate

pip install numpy -U  # Resolve torchmetrics dependencies and update numpy
"""

from _common import *

device_ids = [0, 1, 2, 3]
logger.info(device_ids)
select_device(device_ids)
seed_everything(42)

# ### Loading Model and Tokenizer
BAICHUAN_TYPE = '13B'  # Literal['7B', '13B']
WORK_DIR = f'runs/baichuan_{BAICHUAN_TYPE}'
LORA_TARGET_MODULES = ['W_pack']
#
if BAICHUAN_TYPE == '7B':
    model_id = 'baichuan-inc/baichuan-7B'
    model_dir = get_model_dir(model_id, None)
    model, tokenizer = get_baichuan7B_model_tokenizer(model_dir)
else:
    model_id = 'baichuan-inc/Baichuan-13B-Base'
    model_dir = get_model_dir(model_id, 'v1.0.1')
    model, tokenizer = get_baichuan13B_model_tokenizer(model_dir)
#
GRADIENT_CHECKPOINTING = True
if GRADIENT_CHECKPOINTING:
    # baichuan13B does not implement the `get_input_embeddings` function
    if BAICHUAN_TYPE == '13B':

        def get_input_embeddings(self):
            return self.model.embed_tokens

        model.__class__.get_input_embeddings = get_input_embeddings.__get__(
            model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
#
logger.info(
    f'bos_token_id: {tokenizer.bos_token_id}, eos_token_id: {tokenizer.eos_token_id}, '
    f'pad_token_id: {tokenizer.pad_token_id}')

# ### Preparing lora
LORA_RANK = 8
LORA_ALPHA = 32
LORA_DROPOUT_P = 0.1
lora_config = LoRAConfig(
    replace_modules=LORA_TARGET_MODULES,
    rank=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT_P)
logger.info(f'lora_config: {lora_config}')
Swift.prepare_model(model, lora_config)
#
show_freeze_layers(model)
print_model_info(model)
_p = list(model.parameters())[100]
logger.info(f'device: {_p.device}, dtype: {_p.dtype}')
model.bfloat16()

# ### Loading Dataset
tokenize_function = partial(tokenize_function, tokenizer=tokenizer)
train_dataset, val_dataset = get_alpaca_en_zh_dataset(tokenize_function)
# Data analysis
stat_dataset(train_dataset)
stat_dataset(val_dataset)
data_collate_fn = partial(data_collate_fn, tokenizer=tokenizer)
print_examples(train_dataset[0], tokenizer)

# ### Setting Config
cfg_file = os.path.join(model_dir, 'configuration.json')
#
BATCH_SIZE = 1
MAX_EPOCHS = 1
T_max = get_T_max(len(train_dataset), BATCH_SIZE, MAX_EPOCHS, True)
WORK_DIR = get_work_dir(WORK_DIR)
EVAL_INTERVAL = 500
CONFIG = Config({
    'train': {
        'dataloader': {
            'batch_size_per_gpu': BATCH_SIZE,
            'workers_per_gpu': 1,
            'shuffle': True,
            'drop_last': True,
            'pin_memory': True
        },
        'max_epochs':
        MAX_EPOCHS,
        'work_dir':
        WORK_DIR,
        'optimizer': {
            'type': 'AdamW',
            'lr': 1e-4,
            'weight_decay': 0.01,
            'options': {
                'cumulative_iters': 16,
                'grad_clip': {
                    'norm_type': 2,
                    'max_norm': 2.0
                }
            }
        },
        'lr_scheduler': {
            'type': 'CosineAnnealingLR',
            'T_max': T_max,
            'eta_min': 1e-5,
            'options': {
                'by_epoch': False,
                'warmup': {
                    'type': 'LinearWarmup',
                    'warmup_ratio': 0.1,
                    'warmup_iters': 200
                }
            }
        },
        'hooks': [
            {
                'type': 'CheckpointHook',
                'by_epoch': False,
                'interval': EVAL_INTERVAL,
                'max_checkpoint_num': 1
            },
            {
                'type': 'EvaluationHook',
                'by_epoch': False,
                'interval': EVAL_INTERVAL
            },
            {
                'type': 'BestCkptSaverHook',
                'metric_key': 'acc',
                'save_best': True,
                'rule': 'max',
                'max_checkpoint_num': 1
            },
            {
                'type': 'TextLoggerHook',
                'by_epoch': True,  # Whether EpochBasedTrainer is used
                'interval': 5
            },
            {
                'type': 'TensorboardHook',
                'by_epoch': False,
                'interval': 5
            }
        ]
    },
    'evaluation': {
        'dataloader': {
            'batch_size_per_gpu': BATCH_SIZE,
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
    cfg.update(CONFIG)
    return cfg


trainer = EpochBasedTrainer(
    model=model,
    cfg_file=cfg_file,
    data_collator=data_collate_fn,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    remove_unused_data=True,
    seed=42,
    device='cpu',  # No placement for model, leave the model to `device_map`
    cfg_modify_fn=cfg_modify_fn,
)

trainer.train()

# ### Visualization
tb_dir = os.path.join(WORK_DIR, 'tensorboard_output')
plot_image(tb_dir, ['loss'], 0.9)
