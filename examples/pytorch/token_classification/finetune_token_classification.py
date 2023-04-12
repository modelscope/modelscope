from dataclasses import dataclass, field
from functools import reduce

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.trainers.training_args import TrainingArgs


def reducer(x, y):
    x = x.split(' ') if isinstance(x, str) else x
    y = y.split(' ') if isinstance(y, str) else y
    return x + y


def get_label_list(labels):
    label_enumerate_values = list(set(reduce(reducer, labels)))
    label_enumerate_values.sort()
    return label_enumerate_values


@dataclass
class TokenClassificationArguments(TrainingArgs):

    trainer: str = field(
        default=Trainers.default, metadata={
            'help': 'The trainer used',
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor type',
            'cfg_node': 'preprocessor.type'
        })

    padding: str = field(
        default=None,
        metadata={
            'help': 'The preprocessor padding',
            'cfg_node': 'preprocessor.padding'
        })

    first_sequence: str = field(
        default=None,
        metadata={
            'help': 'The first_sequence of dataset',
            'cfg_node': 'dataset.train.first_sequence'
        })

    label: str = field(
        default=None,
        metadata={
            'help': 'The label of dataset',
            'cfg_node': 'dataset.train.label'
        })
    sequence_length: int = field(
        default=None,
        metadata={
            'help': 'The sequence_length of dataset',
            'cfg_node': 'dataset.train.sequence_length'
        })

    work_dir: str = field(
        default='./tmp',
        metadata={
            'help': 'The working path for saving checkpoint',
        })

    def __call__(self, config):
        config = super().__call__(config)
        if config.safe_get('dataset.train.label') == 'ner_tags':
            label_enumerate_values = get_label_list(undeduplicated_labels)
            config.merge_from_dict(
                {'dataset.train.labels': label_enumerate_values})
        if config.train.lr_scheduler.type == 'LinearLR':
            config.train.lr_scheduler['total_iters'] = \
                int(len(train_dataset) / self.per_device_train_batch_size) * self.max_epochs
        return config


args = TokenClassificationArguments.from_cli(
    task='token-classification', eval_metrics='token-cls-metric')
print(args)

train_dataset = MsDataset.load(
    args.dataset_name,
    subset_name=args.subset_name,
    split='train',
    namespace='damo')['train']
eval_dataset = MsDataset.load(
    args.dataset_name,
    subset_name=args.subset_name,
    split='validation',
    namespace='damo')['validation']

undeduplicated_labels = train_dataset['ner_tags'] + eval_dataset['ner_tags']

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    work_dir=args.work_dir,
    cfg_modify_fn=args)

trainer = build_trainer(name=args.trainer, default_args=kwargs)
trainer.train()
