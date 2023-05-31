import argparse

from dataclasses import dataclass, field
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs

parser = argparse.ArgumentParser()
parser.add_argument("--finetune_mode",
                    type=str,
                    help="dreambooth or lora",
                    default='lora')
args = parser.parse_args()

# choose finetune stable diffusion method, default choice is Lora
if args.finetune_mode == "dreambooth":
    training_args = TrainingArgs(task='diffusers-stable-diffusion').parse_cli()
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
    return cfg

if args.finetune_mode == "dreambooth":
    try:
        kwargs = dict(
            model=training_args.model,
            work_dir=training_args.work_dir,
            model_revision="v1.0.4",
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
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            cfg_modify_fn=cfg_modify_fn_lora)
        trainer: EpochBasedTrainer = build_trainer(name='trainer', default_args=kwargs)
    except Exception as e:
        print(f'Build lora trainer error: {e}')

trainer.train()
