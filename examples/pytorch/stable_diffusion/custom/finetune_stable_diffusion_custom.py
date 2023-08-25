import os
from dataclasses import dataclass, field

import cv2
import torch

from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs
from modelscope.utils.constant import DownloadMode, Tasks


# Load configuration file and dataset
@dataclass(init=False)
class StableDiffusionCustomArguments(TrainingArgs):
    class_prompt: str = field(
        default=None,
        metadata={
            'help':
            'The prompt to specify images in the same class as provided instance images.',
        })

    instance_prompt: str = field(
        default=None,
        metadata={
            'help': 'The prompt with identifier specifying the instance.',
        })

    modifier_token: str = field(
        default=None,
        metadata={
            'help': 'A token to use as a modifier for the concept.',
        })

    num_class_images: int = field(
        default=200,
        metadata={
            'help': 'Minimal class images for prior preservation loss.',
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

    initializer_token: str = field(
        default='ktn+pll+ucd',
        metadata={
            'help': 'A token to use as initializer word.',
        })

    class_data_dir: str = field(
        default='/tmp/class_data',
        metadata={
            'help': 'A folder containing the training data of class images.',
        })

    resolution: int = field(
        default=512,
        metadata={
            'help':
            'The resolution for input images, all the images in the train/validation dataset will be resized to this',
        })

    prior_loss_weight: float = field(
        default=1.0,
        metadata={
            'help': 'The weight of prior preservation loss.',
        })

    freeze_model: str = field(
        default='crossattn_kv',
        metadata={
            'help':
            'crossattn to enable fine-tuning of all params in the cross attention.',
        })

    instance_data_name: str = field(
        default='buptwq/lora-stable-diffusion-finetune-dog',
        metadata={
            'help': 'The instance data local dir or online ID.',
        })

    concepts_list: str = field(
        default=None,
        metadata={
            'help': 'Path to json containing multiple concepts.',
        })

    torch_type: str = field(
        default='float32',
        metadata={
            'help': ' The torch type, default is float32.',
        })


training_args = StableDiffusionCustomArguments(
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
    class_prompt=args.class_prompt,
    instance_prompt=args.instance_prompt,
    modifier_token=args.modifier_token,
    num_class_images=args.num_class_images,
    train_batch_size=args.train_batch_size,
    sample_batch_size=args.sample_batch_size,
    initializer_token=args.initializer_token,
    class_data_dir=args.class_data_dir,
    concepts_list=args.concepts_list,
    resolution=args.resolution,
    prior_loss_weight=args.prior_loss_weight,
    freeze_model=args.freeze_model,
    instance_data_name=args.instance_data_name,
    work_dir=training_args.work_dir,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    torch_type=torch.float16
    if args.torch_type == 'float16' else torch.float32,
    cfg_modify_fn=cfg_modify_fn)

# build trainer and training
trainer = build_trainer(name=Trainers.custom_diffusion, default_args=kwargs)
trainer.train()

# pipeline after training and save result
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model=training_args.model,
    custom_dir=training_args.work_dir + '/output',
    modifier_token=args.modifier_token,
    model_revision=args.model_revision)

output = pipe({'text': args.instance_prompt})
# visualize the result on ipynb and save it
output
cv2.imwrite('./custom_result.png', output['output_imgs'][0])
