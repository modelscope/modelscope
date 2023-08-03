# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from os import path as osp
from typing import Any, Dict

import Image
import open_clip
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from einops import rearrange

from .config import cfg
import modelscope.models.multi_modal.videocomposer.models as models
from modelscope.metainfo import Models
from modelscope.models.base import Model
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.videocomposer.autoencoder import (
    AutoencoderKL, DiagonalGaussianDistribution)
from modelscope.models.multi_modal.videocomposer.clip import (
    FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder)
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

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
        clip_checkpoint = kwargs.pop('clip_checkpoint',
                                     'open_clip_pytorch_model.bin')
        sd_checkpoint = kwargs.pop('sd_checkpoint', 'v2-1_512-ema-pruned.ckpt')
        self.read_image = kwargs.pop('read_image', False)
        self.read_style = kwargs.pop('read_style', True)
        self.read_sketch = kwargs.pop('read_sketch', False)
        self.save_origin_video = kwargs.pop('save_origin_video', True)
        self.video_compositions = kwargs.pop('video_compositions', [
            'text', 'mask', 'depthmap', 'sketch', 'motion', 'image',
            'local_image', 'single_sketch'
        ])
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            layer='penultimate',
            pretrained=os.path.join(model_dir, clip_checkpoint))
        self.clip_encoder = self.clip_encoder.to(self.device)
        self.clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(
            layer='penultimate',
            pretrained=os.path.join(model_dir, clip_checkpoint))
        self.clip_encoder_visual.model.to(self.device)
        self.autoencoder = AutoencoderKL(
            ddconfig, 4, ckpt_path=os.path.join(model_dir, sd_checkpoint))
        self.model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim=cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention=cfg.temporal_attention,
            temporal_attn_times=cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            zero_y=zero_y,
            black_image_feature=black_image_feature,
        ).to(gpu)

    def forward(self, input: Dict[str, Any]):
        # print("--------model input: ", input)
        # input: ref_frame, cap_txt, video_data, misc_data, feature_framerate, mask, mv_data, style_image
        zero_y = self.clip_encoder('').detach()
        black_image_feature = self.clip_encoder_visual(
            self.clip_encoder_visual.black_image).unsqueeze(1)
        black_image_feature = torch.zeros_like(black_image_feature)

        frame_in = None
        if self.read_image:
            image_key = cfg.image_path  #
            frame = Image.open(open(image_key, mode='rb')).convert('RGB')
            frame_in = misc_transforms([frame])

        frame_sketch = None
        if self.read_sketch:
            sketch_key = cfg.sketch_path
            frame_sketch = Image.open(open(sketch_key,
                                           mode='rb')).convert('RGB')
            frame_sketch = misc_transforms([frame_sketch])  #

        frame_style = None
        if self.read_style:
            frame_style = Image.open(open(input['style_image'],
                                          mode='rb')).convert('RGB')

        # Generators for various conditions
        if 'depthmap' in self.video_compositions:
            midas = models.midas_v3(
                pretrained=True).eval().requires_grad_(False).to(
                    memory_format=torch.channels_last).half().to(self.device)
        if 'canny' in self.video_compositions:
            canny_detector = CannyDetector()
        if 'sketch' in self.video_compositions:
            pidinet = pidinet_bsd(
                pretrained=True,
                vanilla_cnn=True).eval().requires_grad_(False).to(self.device)
            cleaner = sketch_simplification_gan(
                pretrained=True).eval().requires_grad_(False).to(self.device)
            pidi_mean = torch.tensor(self.sketch_mean).view(1, -1, 1,
                                                            1).to(self.device)
            pidi_std = torch.tensor(self.sketch_std).view(1, -1, 1,
                                                          1).to(self.device)
        # Placeholder for color inference
        palette = None

        # auotoencoder
        ddconfig = {'double_z': True, 'z_channels': 4, \
                    'resolution': 256, 'in_channels': 3, \
                    'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], \
                    'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0}
        # autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=DOWNLOAD_TO_CACHE(cfg.sd_checkpoint))
        self.autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        self.autoencoder.cuda()

        return video_data.type(torch.float32).cpu()
