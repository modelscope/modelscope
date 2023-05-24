import os
from dataclasses import dataclass, field

from modelscope import (EpochBasedTrainer, MsDataset, TrainingArgs,
                        build_dataset_from_file)
from modelscope.trainers import build_trainer


def set_labels(labels):
    if isinstance(labels, str):
        labels = labels.split(',')
    return {label: id for id, label in enumerate(labels)}


@dataclass(init=False)
class TextClassificationArguments(TrainingArgs):

    first_sequence: str = field(
        default=None,
        metadata={
            'help': 'The first sequence key of preprocessor',
            'cfg_node': 'preprocessor.first_sequence'
        })

    second_sequence: str = field(
        default=None,
        metadata={
            'help': 'The second sequence key of preprocessor',
            'cfg_node': 'preprocessor.second_sequence'
        })

    label: str = field(
        default=None,
        metadata={
            'help': 'The label key of preprocessor',
            'cfg_node': 'preprocessor.label'
        })

    labels: str = field(
        default=None,
        metadata={
            'help': 'The labels of the dataset',
            'cfg_node': 'preprocessor.label2id',
            'cfg_setter': set_labels,
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })


config, args = TextClassificationArguments().parse_cli().to_config()

print(config, args)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.model['num_labels'] = len(cfg.preprocessor.label2id)
    if cfg.train.lr_scheduler.type == 'LinearLR':
        cfg.train.lr_scheduler['total_iters'] = \
            int(len(train_dataset) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
    return cfg


if args.dataset_json_file is None:
    dataset = MsDataset.load(
        args.train_dataset_name, subset_name=args.train_subset_name)
    train_dataset = dataset['train']
    validation_dataset = dataset['validation']
else:
    train_dataset, validation_dataset = build_dataset_from_file(
        args.dataset_json_file)

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    cfg_modify_fn=cfg_modify_fn)

os.environ['LOCAL_RANK'] = str(args.local_rank)
trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
