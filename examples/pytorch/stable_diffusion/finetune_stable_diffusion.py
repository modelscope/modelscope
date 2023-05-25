from dataclasses import dataclass, field

from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs

@dataclass(init=False)
class StableDiffusionArguments(TrainingArgs):
    


config, args = StableDiffusionArguments().parse_cli().to_config()

print(args)

dataset = MsDataset.load(
    args.train_dataset_name, namespace=args.train_dataset_namespace)
train_dataset = dataset['train']
validation_dataset = dataset['validation']


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.train.lr_scheduler.T_max = training_args.max_epochs
    cfg.model.inference = False
    return cfg


kwargs = dict(
    model=training_args.model,
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    cfg_modify_fn=cfg_modify_fn)

trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
