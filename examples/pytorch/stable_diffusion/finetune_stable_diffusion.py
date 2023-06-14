from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs

training_args = TrainingArgs(task='lora-diffusion').parse_cli()
config, args = training_args.to_config()
print(args)

dataset = MsDataset.load(
    args.train_dataset_name, namespace=args.train_dataset_namespace)
train_dataset = dataset['train']
validation_dataset = dataset['validation']


def cfg_modify_fn(cfg):
    cfg.train.max_epochs = args.max_epochs
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': lambda _: 1,
        'last_epoch': -1
    }
    cfg.train.optimizer.lr = 1e-4
    return cfg


kwargs = dict(
    model=training_args.model,
    model_revision=args.model_revision,
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    cfg_modify_fn=cfg_modify_fn)

trainer: EpochBasedTrainer = build_trainer(name=Trainers.lora_diffusion, default_args=kwargs)
trainer.train()
