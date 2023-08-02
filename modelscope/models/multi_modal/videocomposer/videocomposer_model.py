# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from os import path as osp
from typing import Any, Dict

import open_clip
import torch
import torch.cuda.amp as amp
from einops import rearrange

from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.video_synthesis.autoencoder import \
    AutoencoderKL
from modelscope.models.multi_modal.video_synthesis.diffusion import (
    GaussianDiffusion, beta_schedule)
from modelscope.models.multi_modal.video_synthesis.unet_sd import UNetSD
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.models.multi_modal.videocomposer.clip import (FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder)
from modelscope.models.multi_modal.videocomposer.utils import DOWNLOAD_TO_CACHE

__all__ = ['VideoComposer']


@torch.no_grad()
def get_first_stage_encoding(encoder_posterior):
    scale_factor = 0.18215
    if isinstance(encoder_posterior, DiagonalGaussianDistribution):
        z = encoder_posterior.sample()
    elif isinstance(encoder_posterior, torch.Tensor):
        z = encoder_posterior
    else:
        raise NotImplementedError(
            f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented"
        )
    return scale_factor * z


@MODELS.register_module(
    Tasks.text_to_video_synthesis, module_name=Models.videocomposer)
class VideoComposer(Model):
    r"""
    task for video composer.

    Attributes:
        sd_model: denosing model using in this task.
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation into visual space with VQGAN.
        clip_encoder: encode the text into text embedding.
    """

    def __init__(self, model_dir, *args, **kwargs):
        r"""
        Args:
            model_dir (`str` or `os.PathLike`)
                Can be either:
                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co
                      or modelscope.cn. Valid model ids can be located at the root-level, like `bert-base-uncased`,
                      or namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.
                    - A path or url to a *tensorflow index checkpoint file* (e.g, `./tf_model/model.ckpt.index`). In
                      this case, `from_tf` should be set to `True` and a configuration object should be provided as
                      `config` argument. This loading path is slower than converting the TensorFlow checkpoint in a
                      PyTorch model using the provided conversion scripts and loading the PyTorch model afterwards.
                    - A path or url to a model folder containing a *flax checkpoint file* in *.msgpack* format (e.g,
                      `./flax_model/` containing `flax_model.msgpack`). In this case, `from_flax` should be set to
                      `True`.
        """
        super().__init__(model_dir=model_dir, *args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() \
            else torch.device('cpu')
        self.clip_checkpoint = kwargs.pop("clip_checkpoint", 'open_clip_pytorch_model.bin')
        self.clip_encoder = FrozenOpenCLIPEmbedder(layer='penultimate', pretrained = DOWNLOAD_TO_CACHE(self.clip_checkpoint))
        self.clip_encoder = self.clip_encoder.to(self.device)
        self.clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(layer='penultimate',pretrained = DOWNLOAD_TO_CACHE(self.clip_checkpoint))
        self.clip_encoder_visual.model.to(self.device)



    def forward(self, input: Dict[str, Any]):
        print("--------videocomposer model forward input: ", input)
        zero_y = self.clip_encoder("").detach()
        black_image_feature = self.clip_encoder_visual(clip_encoder_visual.black_image).unsqueeze(1)
        black_image_feature = torch.zeros_like(black_image_feature)






        y = input['text_emb']
        zero_y = input['text_emb_zero']
        context = torch.cat([zero_y, y], dim=0).to(self.device)
        # synthesis
        with torch.no_grad():
            num_sample = 1  # here let b = 1
            max_frames = self.config.model.model_args.max_frames
            latent_h, latent_w = input['out_height'] // 8, input[
                'out_width'] // 8
            with amp.autocast(enabled=True):
                x0 = self.diffusion.ddim_sample_loop(
                    noise=torch.randn(num_sample, 4, max_frames, latent_h,
                                      latent_w).to(
                                          self.device),  # shape: b c f h w
                    model=self.sd_model,
                    model_kwargs=[{
                        'y':
                        context[1].unsqueeze(0).repeat(num_sample, 1, 1)
                    }, {
                        'y':
                        context[0].unsqueeze(0).repeat(num_sample, 1, 1)
                    }],
                    guide_scale=9.0,
                    ddim_timesteps=50,
                    eta=0.0)

                scale_factor = 0.18215
                video_data = 1. / scale_factor * x0
                bs_vd = video_data.shape[0]
                video_data = rearrange(video_data, 'b c f h w -> (b f) c h w')
                self.autoencoder.to(self.device)
                video_data = self.autoencoder.decode(video_data)
                if self.config.model.model_args.tiny_gpu == 1:
                    self.autoencoder.to('cpu')
                video_data = rearrange(
                    video_data, '(b f) c h w -> b c f h w', b=bs_vd)
        return video_data.type(torch.float32).cpu()
