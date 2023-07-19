import os
import sys
from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM, AutoTokenizer

from modelscope import (EpochBasedTrainer, MsDataset, TrainingArgs,
                        build_dataset_from_file, snapshot_download)
from modelscope.metainfo import Trainers
from modelscope.preprocessors import TextGenerationTransformersPreprocessor
from modelscope.swift import Swift
from modelscope.swift.lora import LoRAConfig
from modelscope.trainers import build_trainer

DEFAULT_PAD_TOKEN = '[PAD]'
DEFAULT_EOS_TOKEN = '</s>'
DEFAULT_BOS_TOKEN = '<s>'
DEFAULT_UNK_TOKEN = '<unk>'


@dataclass(init=False)
class TextGenerationArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
        })

    src_txt: str = field(
        default=None,
        metadata={
            'help': 'The source text key of preprocessor',
            'cfg_node': 'preprocessor.src_txt'
        })

    tgt_txt: str = field(
        default=None,
        metadata={
            'help': 'The target text key of preprocessor',
            'cfg_node': 'preprocessor.tgt_txt'
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })

    lr_scheduler: str = field(
        default=None,
        metadata={
            'help': 'The lr scheduler type',
            'cfg_node': 'train.lr_scheduler.type'
        })

    world_size: int = field(
        default=None,
        metadata={
            'help': 'The parallel world size',
            'cfg_node': 'megatron.world_size'
        })

    tensor_model_parallel_size: int = field(
        default=None,
        metadata={
            'help': 'The tensor model parallel size',
            'cfg_node': 'megatron.tensor_model_parallel_size'
        })

    use_megatron: bool = field(
        default=None, metadata={
            'help': 'Whether to use MegatronHook',
        })

    bf16: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use bf16',
            'cfg_node': 'train.bf16'
        })

    deepspeed: str = field(
        default=None,
        metadata={
            'help': 'The location of DeepSpeed json config file.',
        })

    T_max: int = field(
        default=None,
        metadata={
            'help': 'The T_max for CosineAnnealingLR',
            'cfg_node': 'train.lr_scheduler.T_max'
        })

    use_lora: int = field(
        default=0,
        metadata={'help': 'Whether to use lora to train the model.'},
    )

    lora_rank: int = field(
        default=32,
        metadata={'help': 'The lora rank'},
    )

    lora_alpha: int = field(
        default=32,
        metadata={'help': 'The lora alpha'},
    )

    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'The lora dropout'},
    )

    device_map: str = field(
        default=None,
        metadata={
            'help': 'A map that specifies where each submodule should go.'
        })


def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer,
                                         model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


config, args = TextGenerationArguments().parse_cli().to_config()
print(config, args)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    if 'hooks' not in cfg.train:
        cfg.train['hooks'] = []
    if args.use_megatron:
        cfg.train.hooks.append({'type': 'MegatronHook'})
    if args.deepspeed:
        cfg.train.hooks.append({
            'type': 'DeepspeedHook',
            'config': args.deepspeed,
            'save_zero_checkpoint': True,
            'with_mpu': False,
        })

    return cfg


if args.dataset_json_file is None:
    train_dataset = MsDataset.load(
        args.train_dataset_name,
        subset_name=args.train_subset_name,
        split=args.train_split,
        namespace=args.train_dataset_namespace)
    validation_dataset = MsDataset.load(
        args.val_dataset_name,
        subset_name=args.val_subset_name,
        split=args.val_split,
        namespace=args.val_dataset_namespace)
else:
    train_dataset, validation_dataset = build_dataset_from_file(
        args.dataset_json_file)

model_dir = snapshot_download(args.model)
sys.path.append(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, trust_remote_code=True, device_map=args.device_map)
cfg_file = os.path.join(model_dir, 'configuration.json')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

special_tokens_dict = dict()
if tokenizer.pad_token is None or tokenizer.pad_token == '':
    special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
if tokenizer.eos_token is None or tokenizer.eos_token == '':
    special_tokens_dict['eos_token'] = DEFAULT_EOS_TOKEN
if tokenizer.bos_token is None or tokenizer.bos_token == '':
    special_tokens_dict['bos_token'] = DEFAULT_BOS_TOKEN
if tokenizer.unk_token is None or tokenizer.unk_token == '':
    special_tokens_dict['unk_token'] = DEFAULT_UNK_TOKEN

smart_tokenizer_and_embedding_resize(
    special_tokens_dict=special_tokens_dict,
    tokenizer=tokenizer,
    model=model,
)

preprocessor = TextGenerationTransformersPreprocessor(
    model_dir,
    tokenizer=tokenizer,
    src_txt=config.preprocessor.src_txt,
    tgt_txt=config.preprocessor.tgt_txt)

if args.use_lora != 0:
    lora_config = LoRAConfig(
        replace_modules=['pack'],
        rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout)
    model = model.bfloat16()
    Swift.prepare_model(model, lora_config)

kwargs = dict(
    model=model,
    cfg_file=cfg_file,
    preprocessor=preprocessor,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    cfg_modify_fn=cfg_modify_fn,
    use_device_map=True)

trainer: EpochBasedTrainer = build_trainer(
    name=args.trainer, default_args=kwargs)
trainer.train()
