from dataclasses import dataclass, field

from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass
class StableDiffusionArguments(TrainingArgs):

    def __call__(self, config):
        config = super().__call__(config)
        config.train.lr_scheduler.T_max = self.max_epochs
        config.model.inference = False
        return config


args = StableDiffusionArguments.from_cli(task='efficient-diffusion-tuning')
print(args)

dataset = MsDataset.load(args.dataset_name, namespace=args.namespace)
train_dataset = dataset['train']
validation_dataset = dataset['validation']

kwargs = dict(
    model=args.model,
    work_dir=args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    cfg_modify_fn=args)

trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()
