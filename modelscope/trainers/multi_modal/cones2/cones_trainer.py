# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms

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
from modelscope.utils.torch_utils import is_dist

PROMPT_TEMPLETE = [
    'a photo of a {}',
    'a rendering of a {}',
    'a cropped photo of the {}',
    'the photo of a {}',
    'a photo of a clean {}',
    'a photo of a dirty {}',
    'a dark photo of the {}',
    'a photo of my {}',
    'a photo of the cool {}',
    'a close-up photo of a {}',
    'a bright photo of the {}',
    'a cropped photo of a {}',
    'a photo of the {}',
    'a good photo of the {}',
    'a photo of one {}',
    'a close-up photo of the {}',
    'a rendition of the {}',
    'a photo of the clean {}',
    'a rendition of a {}',
    'a photo of a nice {}',
    'a good photo of a {}',
    'a photo of the nice {}',
    'a photo of the small {}',
    'a photo of the weird {}',
    'a photo of the large {}',
    'a photo of a cool {}',
    'a photo of a small {}',
]


class ConesCheckpointProcessor(CheckpointProcessor):

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        """Save the state dict for Cones model.
        """
        instance_prompt = 'dog'
        token_num = 1
        pipe = DiffusionPipeline.from_pretrained(self.model_dir, ).to(
            trainer.device)
        text_inputs_origin = pipe.tokenizer(
            instance_prompt,
            padding='max_length',
            max_length=pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors='pt',
        )
        text_inputs_origin_ids = text_inputs_origin.input_ids
        index = text_inputs_origin_ids[0][1]
        prompt_embeds_new = 0
        prompt_embeds_origin = 0
        for template in PROMPT_TEMPLETE:
            text_inputs = pipe.tokenizer(
                template.format(instance_prompt),
                padding='max_length',
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors='pt',
            )
            text_input_ids = text_inputs.input_ids
            index_template = int(torch.where(text_input_ids[0] == index)[0][0])
            prompt_embeds_now = trainer.model.text_encoder(
                text_input_ids.to('cuda'), attention_mask=None)
            prompt_embeds_now = prompt_embeds_now[0][0][
                index_template:index_template + token_num]
            prompt_embeds = pipe.text_encoder(
                text_input_ids.to('cuda'), attention_mask=None)
            prompt_embeds = prompt_embeds[0][0][index_template:index_template
                                                + token_num]
            prompt_embeds_new += prompt_embeds_now
            prompt_embeds_origin += prompt_embeds

        torch.save(
            (prompt_embeds_new - prompt_embeds_origin) / len(PROMPT_TEMPLETE),
            output_dir + '/emb.pt')

        pipeline = DiffusionPipeline.from_pretrained(
            self.model_dir, text_encoder=trainer.model.text_encoder)
        scheduler_args = {}
        pipeline.scheduler = pipeline.scheduler.from_config(
            pipeline.scheduler.config, **scheduler_args)
        pipeline.save_pretrained(output_dir)


@TRAINERS.register_module(module_name=Trainers.cones2_inference)
class ConesDiffusionTrainer(EpochBasedTrainer):

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

        """
        self.with_prior_preservation = kwargs.pop('with_prior_preservation',
                                                  False)
        self.instance_prompt = kwargs.pop('instance_prompt', 'dog')
        self.class_prompt = kwargs.pop('class_prompt', 'a photo of dog')
        self.class_data_dir = kwargs.pop('class_data_dir', '/tmp/class_data')
        self.num_class_images = kwargs.pop('num_class_images', 200)
        self.resolution = kwargs.pop('resolution', 512)
        self.prior_loss_weight = kwargs.pop('prior_loss_weight', 1.0)

        # Save checkpoint and configurate files.
        ckpt_hook = list(
            filter(lambda hook: isinstance(hook, CheckpointHook),
                   self.hooks))[0]
        ckpt_hook.set_processor(ConesCheckpointProcessor(self.model_dir))

        pipeline = DiffusionPipeline.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float32,
            safety_checker=None,
            revision=None,
        )

        pipeline.to(self.device)
        self.target_embed = pipeline.text_encoder(
            pipeline.tokenizer(
                self.instance_prompt,
                truncation=True,
                padding='max_length',
                max_length=pipeline.tokenizer.model_max_length,
                return_tensors='pt',
            ).input_ids.to(self.device))[0].detach()

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.model.text_encoder.parameters(),
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
        token_num = 1
        self.model.text_encoder.train()
        self._mode = ModeKeys.TRAIN
        # call model forward but not __call__ to skip postprocess

        latents = self.model.vae.encode(inputs['target'].to(
            self.device).to(dtype=torch.float32)).latent_dist.sample()
        latents = latents * self.model.vae.config.scaling_factor
        text_inputs = self.model.tokenizer(
            inputs['text'],
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt')
        input_ids = torch.squeeze(text_inputs.input_ids).to(self.device)
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

        encoder_hidden_states = self.model.text_encoder(
            input_ids.unsqueeze(0))[0]

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
        loss_embedding_head = 0.01 * torch.norm(
            torch.squeeze(self.target_embed)[:1]
            - -torch.squeeze(encoder_hidden_states)[:1], 2)
        loss_embedding_tail = 0.001 * torch.norm(
            torch.squeeze(self.target_embed)[1 + token_num:]
            - torch.squeeze(encoder_hidden_states)[1 + token_num:], 2)
        loss_embedding = loss_embedding_head + loss_embedding_tail

        loss = F.mse_loss(
            model_pred_prior.float(), target_prior.float(), reduction='mean')
        # Add the prior loss to the instance loss.
        train_outputs = {OutputKeys.LOSS: loss + loss_embedding}

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
