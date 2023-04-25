from dataclasses import dataclass, field

from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass
class StableDiffusionArguments(TrainingArgs):

    max_epochs: int = field(
        default=None,
        metadata={
            'cfg_node': ['train.max_epochs'],
            'help': 'The max numbers of epochs',
        })

    def __call__(self, config):
        config = super().__call__(config)
        config.train.max_epochs = self.max_epochs
        config.train.lr_scheduler.T_max = self.max_epochs
        config.model.inference = False
        return config


args = StableDiffusionArguments.from_cli(task='efficient-diffusion-tuning')
print(args)

dataset = MsDataset.load(args.dataset_name, namespace='damo')
train_dataset = dataset['train']
validation_dataset = dataset['validation']

kwargs = dict(
    model=args.model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    cfg_modify_fn=args
    )

trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
trainer.train()


