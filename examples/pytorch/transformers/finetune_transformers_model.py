import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          default_data_collator)

from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.default_config import DEFAULT_CONFIG, TrainingArgs


@dataclass
class TransformersArguments(TrainingArgs):

    num_labels: int = field(
        default=None, metadata={
            'help': 'The number of labels',
        })


args = TransformersArguments.from_cli(
    task='text-classification', eval_metrics='seq-cls-metric')

print(args)

dataset = load_dataset(args.dataset_name, args.subset_name)

model = BertForSequenceClassification.from_pretrained(
    args.model, num_labels=args.num_labels)
tokenizer = BertTokenizerFast.from_pretrained(args.model)


def tokenize_sentence(row):
    return tokenizer(row['sentence'], padding='max_length', max_length=128)


# Extra columns, Rename columns
dataset = dataset.map(tokenize_sentence).remove_columns(['sentence',
                                                         'idx']).rename_column(
                                                             'label', 'labels')

cfg_file = os.path.join(args.work_dir or './', 'configuration.json')
DEFAULT_CONFIG.dump(cfg_file)

kwargs = dict(
    model=model,
    cfg_file=cfg_file,
    # data_collator
    data_collator=default_data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['validation'],
    seed=args.seed,
    cfg_modify_fn=args)

os.environ['LOCAL_RANK'] = str(args.local_rank)
trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
