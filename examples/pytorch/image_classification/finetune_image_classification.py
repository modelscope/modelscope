import os
from dataclasses import dataclass, field

from modelscope import MsDataset, TrainingArgs
from modelscope.metainfo import Trainers
from modelscope.trainers.builder import build_trainer


@dataclass(init=False)
class ImageClassificationTrainingArgs(TrainingArgs):
    num_classes: int = field(
        default=None,
        metadata={
            'cfg_node': [
                'model.mm_model.head.num_classes',
                'model.mm_model.train_cfg.augments.0.num_classes',
                'model.mm_model.train_cfg.augments.1.num_classes'
            ],
            'help':
            'number of classes',
        })

    topk: tuple = field(
        default=None,
        metadata={
            'cfg_node': [
                'train.evaluation.metric_options.topk',
                'evaluation.metric_options.topk'
            ],
            'help':
            'evaluation using topk, tuple format, eg (1,), (1,5)',
        })

    warmup_iters: str = field(
        default=None,
        metadata={
            'cfg_node': 'train.lr_config.warmup_iters',
            'help': 'The warmup iters',
        })


def create_dataset(name, split):
    namespace, dataset_name = name.split('/')
    return MsDataset.load(
        dataset_name, namespace=namespace, subset_name='default', split=split)


training_args = ImageClassificationTrainingArgs(
    model='damo/cv_vit-base_image-classification_ImageNet-labels',
    max_epochs=1,
    lr=1e-4,
    optimizer='AdamW',
    warmup_iters=1,
    topk=(1, )).parse_cli()
config, args = training_args.to_config()


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    return cfg


def train():
    train_dataset = create_dataset(
        training_args.train_dataset_name, split=training_args.train_split)
    val_dataset = create_dataset(
        training_args.val_dataset_name, split=training_args.val_split)

    kwargs = dict(
        model=args.model,  # model id
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # validation dataset
        cfg_modify_fn=cfg_modify_fn  # callback to modify configuration
    )

    # in distributed training, specify pytorch launcher
    if 'MASTER_ADDR' in os.environ:
        kwargs['launcher'] = 'pytorch'

    trainer = build_trainer(
        name=Trainers.image_classification, default_args=kwargs)
    # start to train
    trainer.train()


if __name__ == '__main__':
    train()
