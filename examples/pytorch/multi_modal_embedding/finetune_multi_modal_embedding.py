import os
from dataclasses import dataclass, field

from modelscope import MsDataset, TrainingArgs
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.trainers.training_args import set_flatten_value


@dataclass(init=False)
class MultiModalEmbeddingArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
        })

    work_dir: str = field(
        default='./tmp',
        metadata={
            'help': 'The working path for saving checkpoint',
        })

    use_fp16: bool = field(
        default=None,
        metadata={
            'cfg_node': 'train.use_fp16',
            'help': 'Whether to use fp16',
        })

    optimizer_lr: float = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer_hparams.lr',
            'help': 'The learning rate of the optimizer',
        })

    optimizer_hparams: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer_hparams',
            'cfg_setter': set_flatten_value,
            'help': 'The optimizer init params except `lr`',
        })

    loss_aggregate: bool = field(
        default=None,
        metadata={
            'cfg_node': 'train.loss_cfg.aggregate',
            'help': 'Whether to use loss aggregate',
        })

    dataset_column_map: str = field(
        default=None,
        metadata={
            'cfg_node': 'dataset.column_map',
            'cfg_setter': set_flatten_value,
            'help': 'The column map for dataset',
        })

    lr_warmup_proportion: float = field(
        default=None,
        metadata={
            'cfg_node': 'train.lr_scheduler.warmup_proportion',
            'help': 'The warmup proportion for lr scheduler',
        })

    lr_scheduler_hook: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.lr_scheduler_hook',
            'cfg_setter': set_flatten_value,
            'help': 'The parameters for lr scheduler hook',
        })

    optimizer_hook: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer_hook',
            'cfg_setter': set_flatten_value,
            'help': 'The parameters for optimizer hook',
        })

    clip_clamp: bool = field(
        default=None,
        metadata={
            'help': 'Whether to use ClipClampLogitScaleHook',
        })

    world_size: int = field(
        default=None, metadata={
            'help': 'The data parallel world size',
        })


config, args = MultiModalEmbeddingArguments().parse_cli().to_config()
print(config, args)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.merge_from_dict({'pretrained_model.model_name': args.model})
    if args.clip_clamp:
        cfg.train.hooks.append({'type': 'ClipClampLogitScaleHook'})
    if args.world_size > 1:
        cfg.train.launcher = 'pytorch'
    return cfg


train_dataset = MsDataset.load(
    args.train_dataset_name, namespace='modelscope', split='train')
eval_dataset = MsDataset.load(
    args.train_dataset_name, namespace='modelscope', split='validation')

os.makedirs(args.work_dir, exist_ok=True)
kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=args.work_dir,
    cfg_modify_fn=cfg_modify_fn)
trainer = build_trainer(name=args.trainer, default_args=kwargs)
trainer.train()
