import os
from dataclasses import dataclass, field
from functools import reduce
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.trainers.training_args import (TrainingArgs,get_flatten_value,set_flatten_value)

def reducer(x, y):
    x = x.split(' ') if isinstance(x, str) else x
    y = y.split(' ') if isinstance(y, str) else y
    return x + y

def get_label_list(labels):
    label_enumerate_values = list(
        set(reduce(reducer, labels)))
    label_enumerate_values.sort()
    return label_enumerate_values

@dataclass
class TokenClassificationArguments(TrainingArgs):

    name_space:str=field(
        default=None,
        metadata={
            'help': 'The namespace key of dataset',
        })

    train_dataset_params: str = field(
        default=None,
        metadata={
            'cfg_node': 'dataset.train',
            'cfg_getter': get_flatten_value,
            'cfg_setter': set_flatten_value,
            'help': 'The parameters for train dataset',
        })

    preprocessor: str = field(
        default=None,
        metadata={
            'cfg_node': 'preprocessor.type',
            'help': 'The preprocessor type'
        })

    padding: str = field(
        default=None,
        metadata={
            'cfg_node': 'preprocessor.padding',
            'help': 'The preprocessor padding'
        })

    def __call__(self, config):
        config = super().__call__(config)
        if config.safe_get('dataset.train.label') == 'ner_tags':
            label_enumerate_values = get_label_list(train_dataset['ner_tags']+validation_dataset['ner_tags'])
            config.merge_from_dict({'dataset.train.labels':label_enumerate_values})
        if config.train.lr_scheduler.type == 'LinearLR':
            config.train.lr_scheduler['total_iters'] = \
                int(len(train_dataset) / self.per_device_train_batch_size) * self.max_epochs
        return config

args = TokenClassificationArguments.from_cli(
    task='token-classification', eval_metrics='token-cls-metric')

print(args)

train_dataset=MsDataset.load(args.dataset_name, subset_name=args.subset_name,namespace=args.name_space,split='train')['train']
validation_dataset=MsDataset.load(args.dataset_name, subset_name=args.subset_name,namespace=args.name_space,split='validation')['validation']

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    seed=args.seed,
    work_dir=args.work_dir,
    cfg_modify_fn=args)

trainer = build_trainer(name='nlp_base_trainer', default_args=kwargs)
trainer.train()
