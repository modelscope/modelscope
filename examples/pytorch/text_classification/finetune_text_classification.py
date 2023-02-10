import os
from dataclasses import dataclass, field

from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


def get_labels(cfg, metadata):
    label2id = cfg.safe_get(metadata['cfg_node'])
    if label2id is not None:
        return ','.join(label2id.keys())


def set_labels(cfg, labels, metadata):
    if isinstance(labels, str):
        labels = labels.split(',')
    cfg.merge_from_dict(
        {metadata['cfg_node']: {label: id
                                for id, label in enumerate(labels)}})


@dataclass
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
            'cfg_getter': get_labels,
            'cfg_setter': set_labels,
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })

    def __call__(self, config):
        config = super().__call__(config)
        config.model['num_labels'] = len(self.labels)
        if config.train.lr_scheduler.type == 'LinearLR':
            config.train.lr_scheduler['total_iters'] = \
                int(len(train_dataset) / self.per_device_train_batch_size) * self.max_epochs
        return config


args = TextClassificationArguments.from_cli(
    task='text-classification', eval_metrics='seq-cls-metric')

print(args)

dataset = MsDataset.load(args.dataset_name, subset_name=args.subset_name)
train_dataset = dataset['train']
validation_dataset = dataset['validation']

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    cfg_modify_fn=args)

os.environ['LOCAL_RANK'] = str(args.local_rank)
trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
