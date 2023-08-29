# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from modelscope.metainfo import Trainers
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor import \
    CheckpointProcessor
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import ConfigDict


class LoraDiffusionCheckpointProcessor(CheckpointProcessor):

    def __init__(self, torch_type=torch.float32, safe_serialization=False):
        """Checkpoint processor for lora diffusion.

        Args:
            torch_type: The torch type, default is float32.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.

        """
        self.torch_type = torch_type
        self.safe_serialization = safe_serialization

    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None,
                         save_optimizers=True):
        """Save the state dict for lora tune model.
        """
        trainer.model.unet = trainer.model.unet.to(self.torch_type)
        trainer.model.unet.save_attn_procs(
            output_dir, safe_serialization=self.safe_serialization)


@TRAINERS.register_module(module_name=Trainers.lora_diffusion)
class LoraDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """Lora trainers for fine-tuning stable diffusion

        Args:
            lora_rank: The rank size of lora intermediate linear.
            torch_type: The torch type, default is float32.
            safe_serialization: Whether to save the model using safetensors or the traditional PyTorch way with pickle.

        """
        lora_rank = kwargs.pop('lora_rank', 4)
        torch_type = kwargs.pop('torch_type', torch.float32)
        safe_serialization = kwargs.pop('safe_serialization', False)

        # set lora save checkpoint processor
        ckpt_hook = list(
            filter(lambda hook: isinstance(hook, CheckpointHook),
                   self.hooks))[0]
        ckpt_hook.set_processor(
            LoraDiffusionCheckpointProcessor(
                torch_type=torch_type, safe_serialization=safe_serialization))
        # Set correct lora layers
        lora_attn_procs = {}
        for name in self.model.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith(
                'attn1.processor'
            ) else self.model.unet.config.cross_attention_dim
            if name.startswith('mid_block'):
                hidden_size = self.model.unet.config.block_out_channels[-1]
            elif name.startswith('up_blocks'):
                block_id = int(name[len('up_blocks.')])
                hidden_size = list(
                    reversed(
                        self.model.unet.config.block_out_channels))[block_id]
            elif name.startswith('down_blocks'):
                block_id = int(name[len('down_blocks.')])
                hidden_size = self.model.unet.config.block_out_channels[
                    block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=lora_rank)

        self.model.unet.set_attn_processor(lora_attn_procs)

        self.lora_layers = AttnProcsLayers(self.model.unet.attn_processors)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.lora_layers.parameters(),
                cfg=cfg,
                default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e
