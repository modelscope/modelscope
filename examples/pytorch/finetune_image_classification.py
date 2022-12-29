import os
from modelscope.metainfo import Trainers
from modelscope.msdatasets.ms_dataset import MsDataset
from modelscope.trainers.builder import build_trainer
from modelscope.trainers.training_args import ArgAttr, CliArgumentParser, training_args


def define_parser():
    training_args.num_classes = ArgAttr(cfg_node_name=['model.mm_model.head.num_classes',
                                                       'model.mm_model.train_cfg.augments.0.num_classes',
                                                       'model.mm_model.train_cfg.augments.1.num_classes'],
                                        type=int, help='number of classes')

    training_args.train_batch_size.default = 16
    training_args.train_data_worker.default = 1
    training_args.max_epochs.default = 1
    training_args.optimizer.default = 'AdamW'
    training_args.lr.default = 1e-4
    training_args.warmup_iters = ArgAttr('train.lr_config.warmup_iters', type=int, default=1, help='number of warmup epochs')
    training_args.topk = ArgAttr(cfg_node_name=['train.evaluation.metric_options.topk',
                                                'evaluation.metric_options.topk'],
                                 default=(1,), help='evaluation using topk, tuple format, eg (1,), (1,5)')

    training_args.train_data = ArgAttr(type=str, default='tany0699/cats_and_dogs', help='train dataset')
    training_args.validation_data = ArgAttr(type=str, default='tany0699/cats_and_dogs', help='validation dataset')
    training_args.model_id = ArgAttr(type=str, default='damo/cv_vit-base_image-classification_ImageNet-labels', help='model name')

    parser = CliArgumentParser(training_args)
    return parser


def create_dataset(name, split):
    namespace, dataset_name = name.split('/')
    return MsDataset.load(dataset_name, namespace=namespace,
                          subset_name='default',
                          split=split)


def train(parser):
    cfg_dict = parser.get_cfg_dict()
    args = parser.args
    train_dataset = create_dataset(args.train_data, split='train')
    val_dataset = create_dataset(args.validation_data, split='validation')

    def cfg_modify_fn(cfg):
        cfg.merge_from_dict(cfg_dict)
        return cfg

    kwargs = dict(
        model=args.model_id,          # model id
        train_dataset=train_dataset,  # training dataset
        eval_dataset=val_dataset,     # validation dataset
        cfg_modify_fn=cfg_modify_fn     # callback to modify configuration
    )

    # in distributed training, specify pytorch launcher
    if 'MASTER_ADDR' in os.environ:
        kwargs['launcher'] = 'pytorch'

    trainer = build_trainer(name=Trainers.image_classification, default_args=kwargs)
    # start to train
    trainer.train()


if __name__ == '__main__':
    parser = define_parser()
    train(parser)
