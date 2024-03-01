# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

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
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['TextToVideoSynthesis']


@MODELS.register_module(
    Tasks.text_to_video_synthesis, module_name=Models.video_synthesis)
class TextToVideoSynthesis(Model):
    r"""
    task for text to video synthesis.

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
        self.device = torch.device(kwargs.get('device', 'cuda')) if torch.cuda.is_available() \
            else torch.device('cpu')
        self.config = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        cfg = self.config.model.model_cfg
        cfg['temporal_attention'] = True if cfg[
            'temporal_attention'] == 'True' else False

        # Initialize unet
        self.sd_model = UNetSD(
            in_dim=cfg['unet_in_dim'],
            dim=cfg['unet_dim'],
            y_dim=cfg['unet_y_dim'],
            context_dim=cfg['unet_context_dim'],
            out_dim=cfg['unet_out_dim'],
            dim_mult=cfg['unet_dim_mult'],
            num_heads=cfg['unet_num_heads'],
            head_dim=cfg['unet_head_dim'],
            num_res_blocks=cfg['unet_res_blocks'],
            attn_scales=cfg['unet_attn_scales'],
            dropout=cfg['unet_dropout'],
            temporal_attention=cfg['temporal_attention'])
        self.sd_model.load_state_dict(
            torch.load(
                osp.join(model_dir, self.config.model.model_args.ckpt_unet)),
            strict=True)
        self.sd_model.eval()
        self.sd_model.to(self.device)

        # Initialize diffusion
        betas = beta_schedule(
            'linear_sd',
            cfg['num_timesteps'],
            init_beta=0.00085,
            last_beta=0.0120)
        self.diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg['mean_type'],
            var_type=cfg['var_type'],
            loss_type=cfg['loss_type'],
            rescale_timesteps=False)

        # Initialize autoencoder
        ddconfig = {
            'double_z': True,
            'z_channels': 4,
            'resolution': 256,
            'in_channels': 3,
            'out_ch': 3,
            'ch': 128,
            'ch_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attn_resolutions': [],
            'dropout': 0.0
        }
        self.autoencoder = AutoencoderKL(
            ddconfig, 4,
            osp.join(model_dir, self.config.model.model_args.ckpt_autoencoder))
        if self.config.model.model_args.tiny_gpu == 1:
            self.autoencoder.to('cpu')
        else:
            self.autoencoder.to(self.device)
        self.autoencoder.eval()

        # Initialize Open clip
        self.clip_encoder = FrozenOpenCLIPEmbedder(
            version=osp.join(model_dir,
                             self.config.model.model_args.ckpt_clip),
            layer='penultimate')
        if self.config.model.model_args.tiny_gpu == 1:
            self.clip_encoder.to('cpu')
        else:
            self.clip_encoder.to(self.device)

    def forward(self, input: Dict[str, Any]):
        r"""
        The entry function of text to image synthesis task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using vqgan model (autoencoder) to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """
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


class FrozenOpenCLIPEmbedder(torch.nn.Module):
    """
    Uses the OpenCLIP transformer encoder for text
    """
    LAYERS = ['last', 'penultimate']

    def __init__(self,
                 arch='ViT-H-14',
                 version='open_clip_pytorch_model.bin',
                 device='cuda',
                 max_length=77,
                 freeze=True,
                 layer='last'):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device('cpu'), pretrained=version)
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == 'last':
            self.layer_idx = 0
        elif self.layer == 'penultimate':
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
