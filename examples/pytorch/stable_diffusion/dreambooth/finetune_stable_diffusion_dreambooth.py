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
class StableDiffusionDreamboothArguments(TrainingArgs):
    with_prior_preservation: bool = field(
        default=False, metadata={
            'help': 'Whether to enable prior loss.',
        })

    instance_prompt: str = field(
        default='a photo of sks dog',
        metadata={
            'help': 'The instance prompt for dreambooth.',
        })

    class_prompt: str = field(
        default='a photo of dog',
        metadata={
            'help': 'The class prompt for dreambooth.',
        })

    class_data_dir: str = field(
        default='./tmp/class_data',
        metadata={
            'help': 'Save class prompt images path.',
        })

    num_class_images: int = field(
        default=200,
        metadata={
            'help': 'The numbers of saving class images.',
        })

    resolution: int = field(
        default=512, metadata={
            'help': 'The class images resolution.',
        })

    prior_loss_weight: float = field(
        default=1.0,
        metadata={
            'help': 'The weight of instance and prior loss.',
        })

    prompt: str = field(
        default='dog', metadata={
            'help': 'The pipeline prompt.',
        })


training_args = StableDiffusionDreamboothArguments(
    task='text-to-image-synthesis').parse_cli()
config, args = training_args.to_config()

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
    instance_prompt=args.instance_prompt,
    class_prompt=args.class_prompt,
    class_data_dir=args.class_data_dir,
    num_class_images=args.num_class_images,
    resolution=args.resolution,
    prior_loss_weight=args.prior_loss_weight,
    prompt=args.prompt,
    cfg_modify_fn=cfg_modify_fn)

# build trainer and training
trainer = build_trainer(
    name=Trainers.dreambooth_diffusion, default_args=kwargs)
trainer.train()

# pipeline after training and save result
pipe = pipeline(
    task=Tasks.text_to_image_synthesis,
    model=training_args.work_dir + '/output',
    model_revision=args.model_revision)

output = pipe({'text': args.prompt})
cv2.imwrite('./dreambooth_result.png', output['output_imgs'][0])
