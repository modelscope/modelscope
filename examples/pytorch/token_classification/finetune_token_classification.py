from dataclasses import dataclass, field

from modelscope import (EpochBasedTrainer, MsDataset, TrainingArgs,
                        build_dataset_from_file)


@dataclass(init=False)
class TokenClassificationArguments(TrainingArgs):
    trainer: str = field(
        default=None, metadata={
            'help': 'The trainer used',
        })

    work_dir: str = field(
        default='./tmp',
        metadata={
            'help': 'The working path for saving checkpoint',
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })

    preprocessor_padding: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor padding',
            'cfg_node': 'preprocessor.padding'
        })

    mode: str = field(
        default='inference',
        metadata={
            'help': 'The preprocessor padding',
            'cfg_node': 'preprocessor.mode'
        })

    first_sequence: str = field(
        default=None,
        metadata={
            'cfg_node': 'preprocessor.first_sequence',
            'help': 'The parameters for train dataset',
        })

    label: str = field(
        default=None,
        metadata={
            'cfg_node': 'preprocessor.label',
            'help': 'The parameters for train dataset',
        })

    sequence_length: int = field(
        default=128,
        metadata={
            'cfg_node': 'preprocessor.sequence_length',
            'help': 'The parameters for train dataset',
        })


training_args = TokenClassificationArguments().parse_cli()
config, args = training_args.to_config()
print(args)


def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    label_list.sort()
    return label_list


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    labels = train_dataset[training_args.label] + validation_dataset[
        training_args.label]
    label_enumerate_values = get_label_list(labels)
    cfg.merge_from_dict({
        'preprocessor.label2id':
        {label: id
         for id, label in enumerate(label_enumerate_values)}
    })
    cfg.merge_from_dict({'model.num_labels': len(label_enumerate_values)})
    cfg.merge_from_dict({'preprocessor.use_fast': True})
    cfg.merge_from_dict({
        'evaluation.metrics': {
            'type': 'token-cls-metric',
            'label2id':
            {label: id
             for id, label in enumerate(label_enumerate_values)}
        }
    })
    if cfg.train.lr_scheduler.type == 'LinearLR':
        cfg.train.lr_scheduler['total_iters'] = \
            int(len(train_dataset) / cfg.train.dataloader.batch_size_per_gpu) * cfg.train.max_epochs
    return cfg


if args.dataset_json_file is None:
    train_dataset = MsDataset.load(
        args.train_dataset_name,
        subset_name=args.train_subset_name,
        split='train',
        namespace=args.train_dataset_namespace)['train']
    validation_dataset = MsDataset.load(
        args.train_dataset_name,
        subset_name=args.train_subset_name,
        split='validation',
        namespace=args.train_dataset_namespace)['validation']
else:
    train_dataset, validation_dataset = build_dataset_from_file(
        args.dataset_json_file)

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    work_dir=args.work_dir,
    cfg_modify_fn=cfg_modify_fn)

trainer = EpochBasedTrainer(**kwargs)
trainer.train()
