import os
from dataclasses import dataclass, field
from functools import partial

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.trainers.args import (TrainingArgs, get_flatten_value,
                                      set_flatten_value)


@dataclass
class MultiModalEmbeddingArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
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
            'cfg_getter': partial(get_flatten_value, exclusions=['lr']),
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
            'cfg_getter': get_flatten_value,
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
            'cfg_getter': get_flatten_value,
            'cfg_setter': set_flatten_value,
            'help': 'The parameters for lr scheduler hook',
        })

    optimizer_hook: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.optimizer_hook',
            'cfg_getter': get_flatten_value,
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

    def __call__(self, config):
        config = super().__call__(config)
        config.merge_from_dict({'pretrained_model.model_name': self.model})
        if self.clip_clamp:
            config.train.hooks.append({'type': 'ClipClampLogitScaleHook'})
        if self.world_size > 1:
            config.train.launcher = 'pytorch'
        return config


args = MultiModalEmbeddingArguments.from_cli(task='multi-modal-embedding')
print(args)

train_dataset = MsDataset.load(
    args.dataset_name, namespace='modelscope', split='train')
eval_dataset = MsDataset.load(
    args.dataset_name, namespace='modelscope', split='validation')

os.makedirs(args.work_dir, exist_ok=True)
kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=args.work_dir,
    cfg_modify_fn=args)
trainer = build_trainer(name=args.trainer, default_args=kwargs)
trainer.train()
