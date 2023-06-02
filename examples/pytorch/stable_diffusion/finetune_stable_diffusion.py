import sys

from dataclasses import dataclass, field
from modelscope.metainfo import Trainers
from modelscope.msdatasets import MsDataset
from modelscope.trainers import EpochBasedTrainer, build_trainer
from modelscope.trainers.training_args import TrainingArgs


@dataclass(init=False)
class DreamboothDiffusionArguments(TrainingArgs):

    revision: str = field(
        default=None,
        metadata={
            'help': 'The unet and vae revision of stable diffusion.'
        })
    
    model_revision: str = field(
        default='v1.0.5',
        metadata={
            'help': 'The model revision of stable diffusion.'
        })

    prior_loss_weight: float = field(
        default=1.0,
        metadata={
            'help': 'The weight of prior preservation loss.'
        })

    class_prompt: str = field(
        default='a photo of dog',
        metadata={
            'help': 'The prompt to specify images in the same class as provided instance images.'
        })
   
    num_class_images: int = field(
        default=200,
        metadata={
            'help': 'Minimal class images for prior preservation loss.'
        })
    
    sample_batch_size: int = field(
        default=4,
        metadata={
            'help': 'Batch size per device for sampling images.'
        })
    
    center_crop: bool = field(
        default=False,
        metadata={
            'help': 'Whether to center crop the input images to the resolution.'
        })

    tokenizer_max_length: int = field(
        default=None,
        metadata={
            'help': 'The maximum length of the tokenizer.'
        })

    instance_prompt: str = field(
        default="a photo of sks dog",
        metadata={
            'help': 'The prompt with identifier specifying the instance.'
        })

    pretrained_model_name_or_path: str = field(
        default='runwayml/stable-diffusion-v1-5',
        metadata={
            'help': 'The pretrained model name or local path.'
        })
    
    class_data_dir: str = field(
        default='/tmp/class_data',
        metadata={
            'help': 'A folder containing the training data of class images.'
        })

# choose finetune stable diffusion method, default choice is Lora
if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    training_args = DreamboothDiffusionArguments(task='diffusers-stable-diffusion').parse_cli()
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
        'lr_lambda': lambda _: 1,
        'last_epoch': -1
    }                 
    return cfg

if "--finetune_mode" in sys.argv and "dreambooth" in sys.argv:
    try:
        kwargs = dict(
            model=training_args.model,
            work_dir=training_args.work_dir,
            model_revision=args.model_revision,
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            revision=args.revision,
            prior_loss_weight=args.prior_loss_weight,
            class_prompt=args.class_prompt,
            sample_batch_size=args.sample_batch_size,
            center_crop=args.center_crop,
            with_prior_preservation=args.with_prior_preservation,
            tokenizer_max_length=args.tokenizer_max_length,
            instance_prompt=args.instance_prompt,
            class_data_dir=args.class_data_dir,
            num_class_images=args.num_class_images,
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
