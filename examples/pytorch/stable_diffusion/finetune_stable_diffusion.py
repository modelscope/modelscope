import sys

from dataclasses import dataclass, field
from modelscope.metainfo import Trainers
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


# choose finetune stable diffusion method, default choice is Lora
if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    training_args = StableDiffusionDreamboothArguments(task='diffusers-stable-diffusion').parse_cli()
else:
    training_args = TrainingArgs(task='efficient-diffusion-tuning').parse_cli()

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
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': lambda _: 1
    }
    cfg.train.optimizer.lr = 5e-6                    
    return cfg

if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    try:
        kwargs = dict(
            model=training_args.model,
            work_dir=training_args.work_dir,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            cfg_modify_fn=cfg_modify_fn_dreambooth)
        trainer = build_trainer(name=Trainers.dreambooth_diffusion, default_args=kwargs)
    except Exception as e:
        print(f'Build dreambooth trainer error: {e}')
else:
    try:
        kwargs = dict(
            model=training_args.model,
            work_dir=training_args.work_dir,
            instance_prompt=training_args.instance_prompt,
            with_prior_preservation=training_args.with_prior_preservation,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            cfg_modify_fn=cfg_modify_fn_lora)
        trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
    except Exception as e:
        print(f'Build lora trainer error: {e}')

trainer.train()
