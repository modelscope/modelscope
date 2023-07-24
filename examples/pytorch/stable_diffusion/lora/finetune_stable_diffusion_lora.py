import os
from dataclasses import dataclass, field

import cv2

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs
from modelscope.utils.constant import DownloadMode, Tasks


# Load configuration file and dataset
@dataclass(init=False)
class StableDiffusionLoraArguments(TrainingArgs):
    prompt: str = field(
        default='dog', metadata={
            'help': 'The pipeline prompt.',
        })

    lora_rank: int = field(
        default=4,
        metadata={
            'help': 'The rank size of lora intermediate linear.',
        })


training_args = StableDiffusionLoraArguments(
    task='text-to-image-synthesis').parse_cli()
config, args = training_args.to_config()

if os.path.exists(args.train_dataset_name):
    # Load local dataset
    train_dataset = MsDataset.load(args.train_dataset_name)
    validation_dataset = MsDataset.load(args.train_dataset_name)
else:
    # Load online dataset
    train_dataset = MsDataset.load(
        args.train_dataset_name,
        split='train',
        download_mode=DownloadMode.FORCE_REDOWNLOAD)
    validation_dataset = MsDataset.load(
        args.train_dataset_name,
        split='validation',
        download_mode=DownloadMode.FORCE_REDOWNLOAD)


def cfg_modify_fn(cfg):
    if args.use_model_config:
        cfg.merge_from_dict(config)
    else:
        cfg = config
    cfg.train.lr_scheduler = {
        'type': 'LambdaLR',
        'lr_lambda': lambda _: 1,
        'last_epoch': -1
    }
    return cfg


kwargs = dict(
    model=training_args.model,
    model_revision=args.model_revision,
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    lora_rank=args.lora_rank,
    cfg_modify_fn=cfg_modify_fn)

# build trainer and training
trainer = build_trainer(name=Trainers.lora_diffusion, default_args=kwargs)
trainer.train()

# pipeline after training and save result
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model=training_args.model,
    lora_dir=training_args.work_dir + '/output',
    model_revision=args.model_revision)

output = pipe({'text': args.prompt})
# visualize the result on ipynb and save it
output
cv2.imwrite('./lora_result.png', output['output_imgs'][0])
