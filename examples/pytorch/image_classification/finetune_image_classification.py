import os
from dataclasses import dataclass, field

from modelscope.metainfo import Trainers
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.trainers.builder import build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass
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


def train():
    args = ImageClassificationTrainingArgs.from_cli(
        model='damo/cv_vit-base_image-classification_ImageNet-labels',
        max_epochs=1,
        lr=1e-4,
        optimizer='AdamW',
        warmup_iters=1,
        topk=(1, ))
    if args.dataset_name is not None:
        train_dataset = create_dataset(args.dataset_name, split='train')
        val_dataset = create_dataset(args.dataset_name, split='validation')
    else:
        train_dataset = create_dataset(args.train_dataset_name, split='train')
        val_dataset = create_dataset(args.val_dataset_name, split='validation')

    kwargs = dict(
        model=args.model,  # model id
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,  # validation dataset
        cfg_modify_fn=args  # callback to modify configuration
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
