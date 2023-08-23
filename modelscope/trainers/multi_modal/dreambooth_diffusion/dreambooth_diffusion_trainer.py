# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import hashlib
import itertools
import shutil
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from modelscope.metainfo import Trainers
from modelscope.outputs import ModelOutputBase, OutputKeys
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import \
    CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict
from modelscope.utils.constant import ModeKeys
from modelscope.utils.file_utils import func_receive_dict_inputs
from modelscope.utils.torch_utils import is_dist


class DreamboothCheckpointProcessor(CheckpointProcessor):

    def __init__(self,
                 model_dir,
                 torch_type=torch.float32,
                 safe_serialization=False):
        """Checkpoint processor for dreambooth diffusion.

        Args:
            model_dir: The model id or local model dir.
            torch_type: The torch type, default is float32.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.
        """
        self.model_dir = model_dir
        self.torch_type = torch_type
        self.safe_serialization = safe_serialization

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        """Save the state dict for dreambooth model.
        """
        pipeline_args = {}
        if trainer.model.text_encoder is not None:
            pipeline_args['text_encoder'] = trainer.model.text_encoder
        pipeline = DiffusionPipeline.from_pretrained(
            self.model_dir,
            unet=trainer.model.unet,
            torch_type=self.torch_type,
            **pipeline_args,
        )
        scheduler_args = {}
        pipeline.scheduler = pipeline.scheduler.from_config(
            pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(
            output_dir, safe_serialization=self.safe_serialization)


class ClassDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        class_data_root=None,
        class_prompt=None,
        class_num_images=None,
        size=512,
        center_crop=False,
    ):
        """A dataset to prepare  class images with the prompts for fine-tuning the model.
            It pre-processes the images and the tokenizes prompts.

        Args:
            tokenizer: The tokenizer to use for tokenization.
            class_data_root: The saved class data path.
            class_prompt: The prompt to use for class images.
            class_num_images: The number of class images to use.
            size: The size to resize the images.
            center_crop: Whether to do center crop or random crop.

        """
        self.size = size
        self.center_crop = center_crop
        self.tokenizer = tokenizer

        if class_data_root is not None:
            self.class_data_root = Path(class_data_root)
            self.class_data_root.mkdir(parents=True, exist_ok=True)
            self.class_images_path = list(self.class_data_root.iterdir())
            if class_num_images is not None:
                self.num_class_images = min(
                    len(self.class_images_path), class_num_images)
            else:
                self.num_class_images = len(self.class_images_path)
            self.class_prompt = class_prompt
        else:
            raise ValueError(
                f"Class {self.class_data_root} class data root doesn't exists."
            )

        self.image_transforms = transforms.Compose([
            transforms.Resize(
                size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size)
            if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self.num_class_images

    def __getitem__(self, index):
        example = {}

        if self.class_data_root:
            class_image = Image.open(
                self.class_images_path[index % self.num_class_images])
            class_image = exif_transpose(class_image)

            if not class_image.mode == 'RGB':
                class_image = class_image.convert('RGB')
            example['pixel_values'] = self.image_transforms(class_image)

            class_text_inputs = self.tokenizer(
                self.class_prompt,
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt')
            input_ids = torch.squeeze(class_text_inputs.input_ids)
            example['input_ids'] = input_ids

        return example


class PromptDataset(Dataset):

    def __init__(self, prompt, num_samples):
        """Dataset to prepare the prompts to generate class images.

        Args:
            prompt: Class prompt.
            num_samples: The number sample for class images.

        """
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example['prompt'] = self.prompt
        example['index'] = index
        return example


@TRAINERS.register_module(module_name=Trainers.dreambooth_diffusion)
class DreamboothDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Dreambooth trainers for fine-tuning stable diffusion

        Args:
            with_prior_preservation: a boolean indicating whether to enable prior loss.
            instance_prompt: a string specifying the instance prompt.
            class_prompt: a string specifying the class prompt.
            class_data_dir: the path to the class data directory.
            num_class_images: the number of class images to generate.
            prior_loss_weight: the weight of the prior loss.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.

        """
        self.torch_type = kwargs.pop('torch_type', torch.float32)
        self.with_prior_preservation = kwargs.pop('with_prior_preservation',
                                                  False)
        self.instance_prompt = kwargs.pop('instance_prompt',
                                          'a photo of sks dog')
        self.class_prompt = kwargs.pop('class_prompt', 'a photo of dog')
        self.class_data_dir = kwargs.pop('class_data_dir', '/tmp/class_data')
        self.num_class_images = kwargs.pop('num_class_images', 200)
        self.resolution = kwargs.pop('resolution', 512)
        self.prior_loss_weight = kwargs.pop('prior_loss_weight', 1.0)
        safe_serialization = kwargs.pop('safe_serialization', False)

        # Save checkpoint and configurate files.
        ckpt_hook = list(
            filter(lambda hook: isinstance(hook, CheckpointHook),
                   self.hooks))[0]
        ckpt_hook.set_processor(
            DreamboothCheckpointProcessor(
                model_dir=self.model_dir,
                torch_type=self.torch_type,
                safe_serialization=safe_serialization))

        # Check for conflicts and conflicts
        if self.with_prior_preservation:
            if self.class_data_dir is None:
                raise ValueError(
                    'You must specify a data directory for class images.')
            if self.class_prompt is None:
                raise ValueError('You must specify prompt for class images.')
        else:
            if self.class_data_dir is not None:
                warnings.warn(
                    'You need not use --class_data_dir without --with_prior_preservation.'
                )
            if self.class_prompt is not None:
                warnings.warn(
                    'You need not use --class_prompt without --with_prior_preservation.'
                )

        # Generate class images if prior preservation is enabled.
        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                if torch.cuda.device_count() > 1:
                    warnings.warn('Multiple GPU inference not yet supported.')
                pipeline = DiffusionPipeline.from_pretrained(
                    self.model_dir,
                    torch_dtype=self.torch_type,
                    safety_checker=None,
                    revision=None,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images
                sample_dataset = PromptDataset(self.instance_prompt,
                                               num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset)

                pipeline.to(self.device)

                for example in tqdm(
                        sample_dataloader, desc='Generating class images'):
                    images = pipeline(example['prompt']).images
                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Class Dataset and DataLoaders creation
            class_dataset = ClassDataset(
                class_data_root=self.class_data_dir
                if self.with_prior_preservation else None,
                class_prompt=self.class_prompt,
                class_num_images=self.num_class_images,
                tokenizer=self.model.tokenizer,
                size=self.resolution,
                center_crop=False,
            )
            class_dataloader = torch.utils.data.DataLoader(
                class_dataset,
                batch_size=1,
                shuffle=True,
            )
            self.iter_class_dataloader = itertools.cycle(class_dataloader)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.model.unet.parameters(),
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e

    def train_step(self, model, inputs):
        """ Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`TorchModel`): The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        self._mode = ModeKeys.TRAIN
        # call model forward but not __call__ to skip postprocess

        receive_dict_inputs = func_receive_dict_inputs(
            self.unwrap_module(self.model).forward)

        if isinstance(inputs, Mapping) and not receive_dict_inputs:
            train_outputs = model.forward(**inputs)
        else:
            train_outputs = model.forward(inputs)

        if self.with_prior_preservation:
            # Convert to latent space
            batch = next(self.iter_class_dataloader)
            target_prior = batch['pixel_values'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            with torch.no_grad():
                latents = self.model.vae.encode(
                    target_prior.to(
                        dtype=self.torch_type)).latent_dist.sample()
            latents = latents * self.model.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                self.model.noise_scheduler.num_train_timesteps, (bsz, ),
                device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.model.noise_scheduler.add_noise(
                latents, noise, timesteps)

            # Get the text embedding for conditioning
            with torch.no_grad():
                encoder_hidden_states = self.model.text_encoder(input_ids)[0]

            # Get the target for loss depending on the prediction type
            if self.model.noise_scheduler.config.prediction_type == 'epsilon':
                target_prior = noise
            elif self.model.noise_scheduler.config.prediction_type == 'v_prediction':
                target_prior = self.model.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f'Unknown prediction type {self.model.noise_scheduler.config.prediction_type}'
                )

            # Predict the noise residual and compute loss
            model_pred_prior = self.model.unet(noisy_latents, timesteps,
                                               encoder_hidden_states).sample

            # Compute prior loss
            prior_loss = F.mse_loss(
                model_pred_prior.float(),
                target_prior.float(),
                reduction='mean')
            # Add the prior loss to the instance loss.
            train_outputs[
                OutputKeys.LOSS] += self.prior_loss_weight * prior_loss

        if isinstance(train_outputs, ModelOutputBase):
            train_outputs = train_outputs.to_dict()
        if not isinstance(train_outputs, dict):
            raise TypeError('"model.forward()" must return a dict')

        # add model output info to log
        if 'log_vars' not in train_outputs:
            default_keys_pattern = ['loss']
            match_keys = set([])
            for key_p in default_keys_pattern:
                match_keys.update(
                    [key for key in train_outputs.keys() if key_p in key])

            log_vars = {}
            for key in match_keys:
                value = train_outputs.get(key, None)
                if value is not None:
                    if is_dist():
                        value = value.data.clone().to('cuda')
                        dist.all_reduce(value.div_(dist.get_world_size()))
                    log_vars.update({key: value.item()})
            self.log_buffer.update(log_vars)
        else:
            self.log_buffer.update(train_outputs['log_vars'])

        self.train_outputs = train_outputs
