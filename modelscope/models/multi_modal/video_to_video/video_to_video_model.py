# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import random
from copy import copy
from typing import Any, Dict

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

import modelscope.models.multi_modal.video_to_video.utils.transforms as data
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.video_to_video.modules import *
from modelscope.models.multi_modal.video_to_video.modules import (
    AutoencoderKL, FrozenOpenCLIPEmbedder, Vid2VidSDUNet,
    get_first_stage_encoding)
from modelscope.models.multi_modal.video_to_video.utils.config import cfg
from modelscope.models.multi_modal.video_to_video.utils.diffusion_sdedit import \
    GaussianDiffusion_SDEdit
from modelscope.models.multi_modal.video_to_video.utils.schedules_sdedit import \
    noise_schedule
from modelscope.models.multi_modal.video_to_video.utils.seed import setup_seed
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

__all__ = ['VideoToVideo']

logger = get_logger()


@MODELS.register_module(
    Tasks.video_to_video, module_name=Models.video_to_video_model)
class VideoToVideo(TorchModel):
    r"""
    Video2Video aims to solve the task of generating super-resolution videos based on input
    video and text, which is a video generation basic model developed by Alibaba Cloud.

    Paper link: https://arxiv.org/abs/2306.02018

    Attributes:
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation of input video into visual space.
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

        self.config = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))

        cfg.solver_mode = self.config.model.model_args.solver_mode

        # assign default value
        cfg.batch_size = self.config.model.model_cfg.batch_size
        cfg.target_fps = self.config.model.model_cfg.target_fps
        cfg.max_frames = self.config.model.model_cfg.max_frames
        cfg.latent_hei = self.config.model.model_cfg.latent_hei
        cfg.latent_wid = self.config.model.model_cfg.latent_wid
        cfg.model_path = osp.join(model_dir,
                                  self.config.model.model_args.ckpt_unet)

        required_device = kwargs.pop('device', 'gpu')
        self.device = create_device(required_device)

        if 'seed' in self.config.model.model_args.keys():
            cfg.seed = self.config.model.model_args.seed
        else:
            cfg.seed = random.randint(0, 99999)
        setup_seed(cfg.seed)

        # transform
        vid_trans = data.Compose(
            [data.ToTensor(),
             data.Normalize(mean=cfg.mean, std=cfg.std)])
        self.vid_trans = vid_trans

        cfg.embedder.pretrained = osp.join(
            model_dir, self.config.model.model_args.ckpt_clip)
        clip_encoder = FrozenOpenCLIPEmbedder(
            pretrained=cfg.embedder.pretrained, device=self.device)
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f'Build encoder with {cfg.embedder.type}')

        # [unet]
        generator = Vid2VidSDUNet()
        generator = generator.to(self.device)
        generator.eval()
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        ret = generator.load_state_dict(load_dict['state_dict'], strict=True)
        self.generator = generator.half()
        logger.info('Load model {} path {}, with local status {}'.format(
            cfg.UNet.type, cfg.model_path, ret))

        # [diffusion]
        sigmas = noise_schedule(
            schedule='logsnr_cosine_interp',
            n=1000,
            zero_terminal_snr=True,
            scale_min=2.0,
            scale_max=4.0)
        diffusion = GaussianDiffusion_SDEdit(
            sigmas=sigmas, prediction_type='v')
        self.diffusion = diffusion
        logger.info('Build diffusion with type of GaussianDiffusion_SDEdit')

        # [auotoencoder]
        cfg.auto_encoder.pretrained = osp.join(
            model_dir, self.config.model.model_args.ckpt_autoencoder)
        autoencoder = AutoencoderKL(**cfg.auto_encoder)
        autoencoder.eval()
        for param in autoencoder.parameters():
            param.requires_grad = False
        autoencoder.to(self.device)
        self.autoencoder = autoencoder
        torch.cuda.empty_cache()

        negative_prompt = cfg.negative_prompt
        negative_y = clip_encoder(negative_prompt).detach()
        self.negative_y = negative_y

        positive_prompt = cfg.positive_prompt
        self.positive_prompt = positive_prompt

        self.cfg = cfg

    def forward(self, input: Dict[str, Any]):
        r"""
        The entry function of video to video task.
        1. Using CLIP to encode text into embeddings.
        2. Using diffusion model to generate the video's latent representation.
        3. Using autoencoder to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """

        video_data = input['video_data']
        y = input['y']
        cfg = self.cfg

        video_data = F.interpolate(
            video_data, size=(720, 1280), mode='bilinear')
        video_data = video_data.unsqueeze(0)
        video_data = video_data.to(self.device)

        batch_size, frames_num, _, _, _ = video_data.shape
        video_data = rearrange(video_data, 'b f c h w -> (b f) c h w')

        video_data_list = torch.chunk(
            video_data, video_data.shape[0] // 1, dim=0)
        with torch.no_grad():
            decode_data = []
            for vd_data in video_data_list:
                encoder_posterior = self.autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior).detach()
                decode_data.append(tmp)
            video_data_feature = torch.cat(decode_data, dim=0)
            video_data_feature = rearrange(
                video_data_feature, '(b f) c h w -> b c f h w', b=batch_size)
        torch.cuda.empty_cache()

        with amp.autocast(enabled=True):
            total_noise_levels = 600
            t = torch.randint(
                total_noise_levels - 1,
                total_noise_levels, (1, ),
                dtype=torch.long).to(self.device)

            noise = torch.randn_like(video_data_feature)
            noised_lr = self.diffusion.diffuse(video_data_feature, t, noise)
            model_kwargs = [{'y': y}, {'y': self.negative_y}]

            gen_vid = self.diffusion.sample(
                noise=noised_lr,
                model=self.generator,
                model_kwargs=model_kwargs,
                guide_scale=7.5,
                guide_rescale=0.2,
                solver='dpmpp_2m_sde' if cfg.solver_mode == 'fast' else 'heun',
                steps=30 if cfg.solver_mode == 'fast' else 50,
                t_max=total_noise_levels - 1,
                t_min=0,
                discretization='trailing')

            torch.cuda.empty_cache()
            scale_factor = 0.18215
            vid_tensor_feature = 1. / scale_factor * gen_vid

            vid_tensor_feature = rearrange(vid_tensor_feature,
                                           'b c f h w -> (b f) c h w')
            vid_tensor_feature_list = torch.chunk(
                vid_tensor_feature, vid_tensor_feature.shape[0] // 2, dim=0)
            decode_data = []
            for vd_data in vid_tensor_feature_list:
                tmp = self.autoencoder.decode(vd_data)
                decode_data.append(tmp)
            vid_tensor_gen = torch.cat(decode_data, dim=0)

        gen_video = rearrange(
            vid_tensor_gen, '(b f) c h w -> b c f h w', b=cfg.batch_size)

        return gen_video.type(torch.float32).cpu()
