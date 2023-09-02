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
class StableDiffusionCones2Arguments(TrainingArgs):
    instance_prompt: str = field(
        default='a photo of sks dog',
        metadata={
            'help': 'The instance prompt for cones.',
        })

    resolution: int = field(
        default=768, metadata={
            'help': 'The class images resolution.',
        })

    train_batch_size: int = field(
        default=4,
        metadata={
            'help': 'Batch size (per device) for the training dataloader.',
        })

    sample_batch_size: int = field(
        default=4,
        metadata={
            'help': 'Batch size (per device) for sampling images.',
        })

    prompt: str = field(
        default='dog', metadata={
            'help': 'The pipeline prompt.',
        })


training_args = StableDiffusionCones2Arguments(
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
    cfg_modify_fn=cfg_modify_fn)

trainer = build_trainer(name=Trainers.cones2_inference, default_args=kwargs)
trainer.train()

# pipeline after training and save result
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model=training_args.work_dir + '/output',
    model_revision=args.model_revision)

output = pipe({
    'text': 'a mug and a dog on the beach',
    'subject_list': [['mug', 2], ['dog', 5]],
    'color_context': {
        '255,192,0': ['mug', 2.5],
        '255,0,0': ['dog', 2.5]
    },
    'layout': 'data/test/images/mask_example.png'
})
# visualize the result on ipynb and save it
output
cv2.imwrite('./cones2_result.png', output['output_imgs'][0])
