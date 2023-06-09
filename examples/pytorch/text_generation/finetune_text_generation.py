from dataclasses import dataclass, field

from modelscope import (EpochBasedTrainer, MsDataset, TrainingArgs,
                        build_dataset_from_file)
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer


@dataclass(init=False)
class TextGenerationArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
        })

    work_dir: str = field(
        default='./tmp',
        metadata={
            'help': 'The working path for saving checkpoint',
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


def noam_lambda(current_step: int):
    current_step += 1
    return min(current_step**(-0.5), current_step * 100**(-1.5))


config, args = TextGenerationArguments().parse_cli().to_config()
print(config, args)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    if cfg.train.lr_scheduler.type == 'noam':
        cfg.train.lr_scheduler = {
            'type': 'LambdaLR',
            'lr_lambda': noam_lambda,
            'options': {
                'by_epoch': False
            }
        }
    if args.use_megatron:
        cfg.train.hooks.append({'type': 'MegatronHook'})
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

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    work_dir=args.work_dir,
    cfg_modify_fn=cfg_modify_fn)

trainer: EpochBasedTrainer = build_trainer(
    name=args.trainer, default_args=kwargs)
trainer.train()
