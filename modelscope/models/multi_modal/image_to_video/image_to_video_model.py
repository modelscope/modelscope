# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import random
from copy import copy
from typing import Any, Dict

import torch
import torch.cuda.amp as amp

import modelscope.models.multi_modal.image_to_video.utils.transforms as data
from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.image_to_video.modules import *
from modelscope.models.multi_modal.image_to_video.modules import (
    AutoencoderKL, FrozenOpenCLIPVisualEmbedder, Img2VidSDUNet)
from modelscope.models.multi_modal.image_to_video.utils.config import cfg
from modelscope.models.multi_modal.image_to_video.utils.diffusion import \
    GaussianDiffusion
from modelscope.models.multi_modal.image_to_video.utils.seed import setup_seed
from modelscope.models.multi_modal.image_to_video.utils.shedule import \
    beta_schedule
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

__all__ = ['ImageToVideo']

logger = get_logger()


@MODELS.register_module(
    Tasks.image_to_video, module_name=Models.image_to_video_model)
class ImageToVideo(TorchModel):
    r"""
    Image2Video aims to solve the task of generating high-definition videos based on input images.
    Image2Video is a video generation basic model developed by Alibaba Cloud, with a parameter size
    of approximately 2 billion. It has been pre trained on large-scale video and image data and
    fine-tuned on a small amount of high-quality data. The data is widely distributed and diverse
    in categories, and the model has good generalization ability for different types of data

    Paper link: https://arxiv.org/abs/2306.02018

    Attributes:
        diffusion: diffusion model for DDIM.
        autoencoder: decode the latent representation into visual space.
        clip_encoder: encode the image into image embedding.
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
        vid_trans = data.Compose([
            data.CenterCropWide(size=(cfg.resolution[0], cfg.resolution[0])),
            data.Resize(cfg.vit_resolution),
            data.ToTensor(),
            data.Normalize(mean=cfg.vit_mean, std=cfg.vit_std)
        ])
        self.vid_trans = vid_trans

        cfg.embedder.pretrained = osp.join(
            model_dir, self.config.model.model_args.ckpt_clip)
        clip_encoder = FrozenOpenCLIPVisualEmbedder(
            device=self.device, **cfg.embedder)
        clip_encoder.model.to(self.device)
        self.clip_encoder = clip_encoder
        logger.info(f'Build encoder with {cfg.embedder.type}')

        # [unet]
        generator = Img2VidSDUNet(**cfg.UNet)
        generator = generator.to(self.device)
        generator.eval()
        load_dict = torch.load(cfg.model_path, map_location='cpu')
        ret = generator.load_state_dict(load_dict['state_dict'], strict=True)
        self.generator = generator
        logger.info('Load model {} path {}, with local status {}'.format(
            cfg.UNet.type, cfg.model_path, ret))

        # [diffusion]
        betas = beta_schedule(
            'linear_sd',
            cfg.num_timesteps,
            init_beta=0.00085,
            last_beta=0.0120)
        diffusion = GaussianDiffusion(
            betas=betas,
            mean_type=cfg.mean_type,
            var_type=cfg.var_type,
            loss_type=cfg.loss_type,
            rescale_timesteps=False,
            noise_strength=getattr(cfg, 'noise_strength', 0))
        self.diffusion = diffusion
        logger.info('Build diffusion with type of GaussianDiffusion')

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

        zero_feature = torch.zeros(1, 1, cfg.UNet.input_dim).to(self.device)
        self.zero_feature = zero_feature
        self.fps_tensor = torch.tensor([cfg.target_fps],
                                       dtype=torch.long,
                                       device=self.device)
        self.cfg = cfg

    def forward(self, input: Dict[str, Any]):
        r"""
        The entry function of image to video task.
        1. Using diffusion model to generate the video's latent representation.
        2. Using autoencoder to decode the video's latent representation to visual space.

        Args:
            input (`Dict[Str, Any]`):
                The input of the task
        Returns:
            A generated video (as pytorch tensor).
        """

        vit_frame = input['vit_frame']
        cfg = self.cfg

        img_embedding = self.clip_encoder(vit_frame).unsqueeze(1)

        noise = self.build_noise()
        zero_feature = copy(self.zero_feature)
        with torch.no_grad():
            with amp.autocast(enabled=cfg.use_fp16):
                model_kwargs = [{
                    'y': img_embedding,
                    'fps': self.fps_tensor
                }, {
                    'y': zero_feature.repeat(cfg.batch_size, 1, 1),
                    'fps': self.fps_tensor
                }]
                gen_video = self.diffusion.ddim_sample_loop(
                    noise=noise,
                    model=self.generator,
                    model_kwargs=model_kwargs,
                    guide_scale=cfg.guide_scale,
                    ddim_timesteps=cfg.ddim_timesteps,
                    eta=0.0)

            gen_video = 1. / cfg.scale_factor * gen_video
            gen_video = rearrange(gen_video, 'b c f h w -> (b f) c h w')
            chunk_size = min(cfg.decoder_bs, gen_video.shape[0])
            gen_video_list = torch.chunk(
                gen_video, gen_video.shape[0] // chunk_size, dim=0)
            decode_generator = []
            for vd_data in gen_video_list:
                gen_frames = self.autoencoder.decode(vd_data)
                decode_generator.append(gen_frames)

        gen_video = torch.cat(decode_generator, dim=0)
        gen_video = rearrange(
            gen_video, '(b f) c h w -> b c f h w', b=cfg.batch_size)

        return gen_video.type(torch.float32).cpu()

    def build_noise(self):
        cfg = self.cfg
        noise = torch.randn(
            [1, 4, cfg.max_frames, cfg.latent_hei,
             cfg.latent_wid]).to(self.device)
        if cfg.noise_strength > 0:
            b, c, f, *_ = noise.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=noise.device)
            noise = noise + cfg.noise_strength * offset_noise
        return noise.contiguous()
