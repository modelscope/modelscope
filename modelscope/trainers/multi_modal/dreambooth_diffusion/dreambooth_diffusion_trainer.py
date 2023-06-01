# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union
from collections.abc import Mapping

import os
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributed as dist
from torch.utils.data import Dataset
import hashlib
import random
from PIL import Image
from PIL.ImageOps import exif_transpose
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers import (AutoencoderKL, DDPMScheduler,
                       UNet2DConditionModel, DiffusionPipeline)
from torchvision import transforms
from transformers import AutoTokenizer, PretrainedConfig
from modelscope.metainfo import Trainers
from modelscope.outputs import OutputKeys
from modelscope.models.base import TorchModel
from modelscope.trainers.hooks.checkpoint import CheckpointProcessor
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.torch_utils import is_dist
from modelscope.utils.constant import ModeKeys
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.checkpoint import (save_checkpoint, save_configuration,
                                         save_pretrained)

class UnetModel(TorchModel):
    def __init__(self, unet, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = unet
    
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)



class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class DreamboothCheckpointProcessor(CheckpointProcessor):

    @staticmethod
    def _bin_file(model):
        """Get bin file path for diffuser.
        """
        default_bin_file = 'diffusion_pytorch_model.bin'
        return default_bin_file


@TRAINERS.register_module(module_name=Trainers.dreambooth_diffusion)
class DreamboothDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        self.revision = kwargs.pop("revision", None)
        self.prior_loss_weight = kwargs.pop('prior_loss_weight', 1.0)
        self.class_prompt = kwargs.pop('class_prompt', "a photo of dog")
        self.sample_batch_size = kwargs.pop('sample_batch_size', 4)
        self.center_crop = kwargs.pop('center_crop', False)
        self.with_prior_preservation = kwargs.pop('with_prior_preservation', False)
        self.tokenizer_max_length = kwargs.pop('tokenizer_max_length', None)
        self.instance_prompt = kwargs.pop('instance_prompt', "a photo of sks dog")
        self.pretrained_model_name_or_path = kwargs.pop('pretrained_model_name_or_path', 
                                                        "runwayml/stable-diffusion-v1-5")
        self.class_data_dir = kwargs.pop("class_data_dir", "/tmp/class_data")
        self.num_class_images = kwargs.pop("num_class_images", 200)
        super().__init__(*args, **kwargs)
        print("-------self.hooks: ", self.hooks)
        ckpt_hook = filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks)[0]
        ckpt_hook.set_processor(DreamboothCheckpointProcessor())

    def build_model(self) -> Union[nn.Module, TorchModel]:
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.pretrained_model_name_or_path, subfolder='scheduler')
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='tokenizer',
            revision=self.revision,
            use_fast=False)
        self.vae = AutoencoderKL.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='vae',
            revision=self.revision)
        self.unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path,
            subfolder='unet',
            revision=self.revision)
        # import correct text encoder class
        text_encoder_cls = self.import_model_class_from_model_name_or_path(self.pretrained_model_name_or_path)
        self.text_encoder = text_encoder_cls.from_pretrained(
            self.pretrained_model_name_or_path, 
            subfolder="text_encoder", 
            revision=self.revision)
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        
        return UnetModel(self.unet)

    def train(self,
              checkpoint_path=None,
              load_all_state=True,
              *args,
              **kwargs):
        """Start training.
        Args:
            checkpoint_path(`str`, `optional`): The previous saving checkpoint to read,
                usually it's a `some-file-name.pth` file generated by this trainer.
            load_all_state(`bool`: `optional`): Load all state out of the `checkpoint_path` file, including the
                state dict of model, optimizer, lr_scheduler, the random state and epoch/iter number. If False, only
                the model's state dict will be read, and model will be trained again.
            kwargs:
                strict(`boolean`): If strict, any unmatched keys will cause an error.
        """
        # Generate class images if prior preservation is enabled.
        if self.with_prior_preservation:
            class_images_dir = Path(self.class_data_dir)
            if not class_images_dir.exists():
                class_images_dir.mkdir(parents=True)
            cur_class_images = len(list(class_images_dir.iterdir()))

            if cur_class_images < self.num_class_images:
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    revision=self.revision,
                )
                pipeline.set_progress_bar_config(disable=True)

                num_new_images = self.num_class_images - cur_class_images
                sample_dataset = PromptDataset(self.class_prompt, num_new_images)
                sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=self.sample_batch_size)

                pipeline.to(self.device)

                for example in tqdm(sample_dataloader, desc="Generating class images"):
                    images = pipeline(example["prompt"]).images
                    for i, image in enumerate(images):
                        hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                        image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                        image.save(image_filename)

                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self._mode = ModeKeys.TRAIN
        self.train_dataloader = self.get_train_dataloader()
        self.data_loader = self.train_dataloader
        self.register_optimizers_hook()
        self.register_processors()
        self.print_hook_info()
        self.set_checkpoint_file_to_hook(checkpoint_path, load_all_state,
                                         kwargs.get('strict', False))
        self.model.train()
        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
        self.text_encoder = self.text_encoder.to(self.device)

        self.train_loop(self.train_dataloader)

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

        # inputs
        batch = self.data_assemble(inputs)
        pixel_values = batch["pixel_values"].to(dtype=torch.float32)
        if self.vae is not None:
            # Convert images to latent space
            model_input = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
        else:
            model_input = pixel_values

        noise = torch.randn_like(model_input)
        bsz = model_input.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
        timesteps = timesteps.long()
        # Add noise to the model input according to the noise magnitude at each timestep
        noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
        # Get the text embedding for conditioning
        encoder_hidden_states = self.encode_prompt(
            self.text_encoder,
            batch["input_ids"],
            batch["attention_mask"],
            text_encoder_use_attention_mask=False,
        )

        # Predict the noise residual by unet model
        model_pred = model(noisy_model_input, timesteps, encoder_hidden_states).sample
        if model_pred.shape[1] == 6:
            model_pred, _ = torch.chunk(model_pred, 2, dim=1)
        
        # Get the target for loss depending on the prediction type
        if self.noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif self.noise_scheduler.config.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
        if self.with_prior_preservation:
            # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
            model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
            target, target_prior = torch.chunk(target, 2, dim=0)
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            # Compute prior loss
            prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
            # Add the prior loss to the instance loss.
            loss = loss + self.prior_loss_weight * prior_loss
        else:
            # Compute instance loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        train_outputs = {OutputKeys.LOSS: loss}
        print("loss: ", loss)

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

    def evaluation_step(self, data):
        self.model.eval()

        with torch.no_grad():
            batch = self.data_assemble(data)
            model_input = self.vae.encode(batch["pixel_values"]).latent_dist.sample()
            model_input = model_input * self.vae.config.scaling_factor
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]
            timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=model_input.device)
            timesteps = timesteps.long()
            noisy_model_input = self.noise_scheduler.add_noise(model_input, noise, timesteps)
            encoder_hidden_states = self.encode_prompt(
                self.text_encoder,
                batch["input_ids"],
                batch["attention_mask"],
                text_encoder_use_attention_mask=False,
            )

            # Predict reslut by unet model
            model_pred = self.model(noisy_model_input, timesteps, encoder_hidden_states).sample
            if model_pred.shape[1] == 6:
                model_pred, _ = torch.chunk(model_pred, 2, dim=1)
            
            # Get the target for loss depending on the prediction type
            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(model_input, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")
            
            if self.with_prior_preservation:
                # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                target, target_prior = torch.chunk(target, 2, dim=0)
                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                # Compute prior loss
                prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")
                # Add the prior loss to the instance loss.
                loss = loss + self.prior_loss_weight * prior_loss
            else:
                # Compute instance loss
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            
            result = {OutputKeys.LOSS: loss}
    
        return result

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     return self.model.state_dict(destination, prefix, keep_vars)

    # def load_state_dict(self,
    #                     state_dict: 'OrderedDict[str, Tensor]',
    #                     strict: bool = True):
    #     return self.model.load_state_dict(state_dict, strict)

    def import_model_class_from_model_name_or_path(self, pretrained_model_name_or_path: str):
        text_encoder_config = PretrainedConfig.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=self.revision
        )
        model_class = text_encoder_config.architectures[0]

        if model_class == "CLIPTextModel":
            from transformers import CLIPTextModel

            return CLIPTextModel
        elif model_class == "RobertaSeriesModelWithTransformation":
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

            return RobertaSeriesModelWithTransformation
        elif model_class == "T5EncoderModel":
            from transformers import T5EncoderModel

            return T5EncoderModel
        else:
            raise ValueError(f"{model_class} is not supported.")

    def encode_prompt(self, text_encoder, input_ids, attention_mask, text_encoder_use_attention_mask=None):
        text_input_ids = input_ids.to(text_encoder.device)

        if text_encoder_use_attention_mask:
            attention_mask = attention_mask.to(text_encoder.device)
        else:
            attention_mask = None

        prompt_embeds = text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        return prompt_embeds
    
    def data_assemble(self, inputs):
        batch = {}
        batch["pixel_values"] = inputs["target"]
        text_inputs = self.tokenize_prompt(self.tokenizer, self.instance_prompt)
        batch["input_ids"] = text_inputs.input_ids
        batch["attention_mask"] = text_inputs.attention_mask

        # prepare prior loss dataset
        if self.with_prior_preservation: 
            if self.class_data_dir is not None:
                image_transforms = transforms.Compose(
                    [
                        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
                        transforms.CenterCrop(512) if self.center_crop else transforms.RandomCrop(512),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5], [0.5]),
                    ]
                )
                self.class_data_root = Path(self.class_data_dir)
                self.class_data_root.mkdir(parents=True, exist_ok=True)
                self.class_images_path = list(self.class_data_root.iterdir())
                if self.num_class_images is not None:
                    self.num_class_images = min(len(self.class_images_path), self.num_class_images)
                else:
                    self.num_class_images = len(self.class_images_path)
                self.class_prompt = self.class_prompt

                index = random.randint(0, self.num_class_images)
                class_image = Image.open(self.class_images_path[index % self.num_class_images])
                class_image = exif_transpose(class_image)

                if not class_image.mode == "RGB":
                    class_image = class_image.convert("RGB")
                
                pixel_values = []
                for pixel_value in batch["pixel_values"]:
                    pixel_values.append(pixel_value)
                pixel_values.append(image_transforms(class_image).to(self.device))
                batch["pixel_values"] = torch.stack(pixel_values)

                class_text_inputs = self.tokenize_prompt(
                    self.tokenizer, self.class_prompt, tokenizer_max_length=self.tokenizer_max_length
                )
                input_ids = class_text_inputs.input_ids
                batch["input_ids"] = torch.cat((batch["input_ids"], input_ids), dim=0)

                batch["attention_mask"] += class_text_inputs.attention_mask
            else:
                raise ValueError(
                    f"Prepare prior preservation dataset error."
                    f"Parameter with_prior_preservation is True but class_data_dir is None")
        
        return batch

    def tokenize_prompt(self, tokenizer, prompt, tokenizer_max_length=None):
        if tokenizer_max_length is not None:
            max_length = tokenizer_max_length
        else:
            max_length = tokenizer.model_max_length

        text_inputs = tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        return text_inputs