import os
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (BertForSequenceClassification, BertTokenizerFast,
                          default_data_collator)

from modelscope import TrainingArgs
from modelscope.trainers import EpochBasedTrainer, build_trainer


@dataclass(init=False)
class TransformersArguments(TrainingArgs):

    num_labels: int = field(
        default=None, metadata={
            'help': 'The number of labels',
        })

    sentence: str = field(
        default=None, metadata={
            'help': 'The sentence key',
        })

    label: str = field(
        default=None, metadata={
            'help': 'The label key',
        })


training_args = TransformersArguments(
    task='text-classification', eval_metrics='seq-cls-metric').parse_cli()
config, args = training_args.to_config()

print(config, args)

train_dataset = load_dataset(
    args.train_dataset_name, args.train_subset_name, split=args.train_split)
val_dataset = load_dataset(
    args.val_dataset_name, args.val_subset_name, split=args.val_split)

model = BertForSequenceClassification.from_pretrained(
    args.model, num_labels=args.num_labels)
tokenizer = BertTokenizerFast.from_pretrained(args.model)


def tokenize_sentence(row):
    return tokenizer(
        row[training_args.sentence], padding='max_length', max_length=128)


# Extra columns, Rename columns
train_dataset = train_dataset.map(tokenize_sentence)
val_dataset = val_dataset.map(tokenize_sentence)
if training_args.label != 'labels':
    train_dataset = train_dataset.rename_columns(
        {training_args.label: 'labels'})
    val_dataset = val_dataset.rename_columns({training_args.label: 'labels'})

cfg_file = os.path.join(args.work_dir or './', 'configuration.json')
config.dump(cfg_file)

kwargs = dict(
    model=model,
    cfg_file=cfg_file,
    # data_collator
    data_collator=default_data_collator,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    remove_unused_data=True,
    seed=args.seed)

os.environ['LOCAL_RANK'] = str(args.local_rank)
trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
