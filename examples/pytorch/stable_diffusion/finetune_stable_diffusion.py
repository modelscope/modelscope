import sys

from dataclasses import dataclass, field
from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass
class StableDiffusionDreamboothArguments(TrainingArgs):

    pretrained_model_name_or_path: str = field(
        default="runwayml/stable-diffusion-v1-5",
        metadata={
            'help': 'The pretrained model of stable diffusion',
            'cfg_node': 'model.pretrained_model_name_or_path'
        })

@dataclass
class StableDiffusionLoraArguments(TrainingArgs):

    def __call__(self, config):
        config = super().__call__(config)
        config.train.lr_scheduler.T_max = self.max_epochs
        config.model.inference = False
        return config

# choose finetune stable diffusion method, default choice is Lora
if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    training_args = StableDiffusionDreamboothArguments(task='diffusers-stable-diffusion').parse_cli()
else:
    training_args = StableDiffusionLoraArguments(task='efficient-diffusion-tuning').parse_cli()

config, args = training_args.to_config()
print(args)

dataset = MsDataset.load(
    args.train_dataset_name, namespace=args.train_dataset_namespace)
train_dataset = dataset['train']
validation_dataset = dataset['validation']

def cfg_modify_fn_lora(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.model.inference = False
    return cfg

def cfg_modify_fn_dreambooth(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    return cfg

if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    kwargs = dict(
        model=training_args.model,
        work_dir=training_args.work_dir,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        cfg_modify_fn=cfg_modify_fn_dreambooth)
else:
    kwargs = dict(
        model=training_args.model,
        work_dir=training_args.work_dir,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        cfg_modify_fn=cfg_modify_fn_lora)
    trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)

trainer.train()
