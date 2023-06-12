# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
from typing import Union

import torch
from torch import nn
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor

from modelscope.metainfo import Trainers
from modelscope.models.base import Model, TorchModel
from modelscope.trainers.builder import TRAINERS
from modelscope.trainers.optimizer.builder import build_optimizer
from modelscope.trainers.trainer import EpochBasedTrainer
from modelscope.utils.config import Config, ConfigDict
from modelscope.trainers.hooks.checkpoint.checkpoint_hook import \
    CheckpointHook
from modelscope.trainers.hooks.checkpoint.checkpoint_processor \
    import CheckpointProcessor


class LoraDiffusionCheckpointProcessor(CheckpointProcessor):

    # @staticmethod
    # def _bin_file(model):
    #     """Get bin file path for diffuser.
    #     """
    #     default_bin_file = 'pytorch_lora_weights.bin'
    #     return default_bin_file
    def save_checkpoints(self,
                         trainer,
                         checkpoint_path_prefix,
                         output_dir,
                         meta=None):
        """Save the state dict for trainer and model.

        This is a strategic function which can be registered by other hook's function.

        Args:
            trainer(`EpochBasedTrainer`): The trainer instance.
            checkpoint_path_prefix(`str`): The saving dir with a prefix.
                like: /tmp/test/epoch_0
            output_dir(`str`): The output dir for inference.
            meta: (`dict`): The meta info needed to be saved into files.
        """
        # model = trainer.unwrap_module(trainer.model)
        # _model_file, _train_state_file = self._get_state_file_name(
        #     checkpoint_path_prefix)

        # # Save pth file without model state_dict
        # self.save_trainer_state(trainer, model, _train_state_file, meta)
        # self.save_model_state(model, _model_file)
        # self.link(model, _model_file, output_dir)
        print("--------save checkpoints output_dir: ", output_dir)
        self.model.unet = self.model.unet.to(torch.float32)
        self.model.unet.save_attn_procs(output_dir)
    

@TRAINERS.register_module(module_name=Trainers.lora_diffusion)
class LoraDiffusionTrainer(EpochBasedTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ckpt_hook = list(filter(lambda hook: isinstance(hook, CheckpointHook), self.hooks))[0]
        ckpt_hook.set_processor(LoraDiffusionCheckpointProcessor())
        # Set correct lora layers
        lora_attn_procs = {}
        for name in self.model.unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.model.unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.model.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.model.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.model.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)

        self.model.unet.set_attn_processor(lora_attn_procs)
        
        self.lora_layers = AttnProcsLayers(self.model.unet.attn_processors)

    def build_optimizer(self, cfg: ConfigDict, default_args: dict = None):
        try:
            return build_optimizer(
                self.lora_layers.parameters(), cfg=cfg, default_args=default_args)
        except KeyError as e:
            self.logger.error(
                f'Build optimizer error, the optimizer {cfg} is a torch native component, '
                f'please check if your torch with version: {torch.__version__} matches the config.'
            )
            raise e
        