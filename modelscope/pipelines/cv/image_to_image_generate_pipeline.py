# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image

import modelscope.models.cv.image_to_image_generation.data as data
import modelscope.models.cv.image_to_image_generation.models as models
import modelscope.models.cv.image_to_image_generation.ops as ops
from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_to_image_generation.model import UNet
from modelscope.models.cv.image_to_image_generation.models.clip import \
    VisionTransformer
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_to_image_generation,
    module_name=Pipelines.image_to_image_generation)
class Image2ImageGenerationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image-to-image generation pipeline
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.repetition = 4
        # load vit model
        vit_model_path = osp.join(self.model,
                                  self.cfg.ModelPath.vit_model_path)
        logger.info(f'loading vit model from {vit_model_path}')
        self.vit = VisionTransformer(
            image_size=self.cfg.Params.vit.vit_image_size,
            patch_size=self.cfg.Params.vit.vit_patch_size,
            dim=self.cfg.Params.vit.vit_dim,
            out_dim=self.cfg.Params.vit.vit_out_dim,
            num_heads=self.cfg.Params.vit.vit_num_heads,
            num_layers=self.cfg.Params.vit.vit_num_layers).eval(
            ).requires_grad_(False).to(self._device)  # noqa E123
        state = torch.load(vit_model_path)
        state = {
            k[len('visual.'):]: v
            for k, v in state.items() if k.startswith('visual.')
        }
        self.vit.load_state_dict(state)
        logger.info('load vit model done')

        # load autoencoder model
        ae_model_path = osp.join(self.model, self.cfg.ModelPath.ae_model_path)
        logger.info(f'loading autoencoder model from {ae_model_path}')
        self.autoencoder = models.VQAutoencoder(
            dim=self.cfg.Params.ae.ae_dim,
            z_dim=self.cfg.Params.ae.ae_z_dim,
            dim_mult=self.cfg.Params.ae.ae_dim_mult,
            attn_scales=self.cfg.Params.ae.ae_attn_scales,
            codebook_size=self.cfg.Params.ae.ae_codebook_size).eval(
            ).requires_grad_(False).to(self._device)  # noqa E123
        self.autoencoder.load_state_dict(
            torch.load(ae_model_path, map_location=self._device))
        logger.info('load autoencoder model done')

        # load decoder model
        decoder_model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading decoder model from {decoder_model_path}')
        self.decoder = UNet(
            resolution=self.cfg.Params.unet.unet_resolution,
            in_dim=self.cfg.Params.unet.unet_in_dim,
            dim=self.cfg.Params.unet.unet_dim,
            label_dim=self.cfg.Params.vit.vit_out_dim,
            context_dim=self.cfg.Params.unet.unet_context_dim,
            out_dim=self.cfg.Params.unet.unet_out_dim,
            dim_mult=self.cfg.Params.unet.unet_dim_mult,
            num_heads=self.cfg.Params.unet.unet_num_heads,
            head_dim=None,
            num_res_blocks=self.cfg.Params.unet.unet_res_blocks,
            attn_scales=self.cfg.Params.unet.unet_attn_scales,
            dropout=self.cfg.Params.unet.unet_dropout).eval().requires_grad_(
                False).to(self._device)
        self.decoder.load_state_dict(
            torch.load(decoder_model_path, map_location=self._device))
        logger.info('load decoder model done')

        # diffusion
        logger.info('Initialization diffusion ...')
        betas = ops.beta_schedule(self.cfg.Params.diffusion.schedule,
                                  self.cfg.Params.diffusion.num_timesteps)
        self.diffusion = ops.GaussianDiffusion(
            betas=betas,
            mean_type=self.cfg.Params.diffusion.mean_type,
            var_type=self.cfg.Params.diffusion.var_type,
            loss_type=self.cfg.Params.diffusion.loss_type,
            rescale_timesteps=False)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        input_img_list = []
        if isinstance(input, str):
            input_img_list = [input]
            input_type = 0
        elif isinstance(input, tuple) and len(input) == 2:
            input_img_list = list(input)
            input_type = 1
        else:
            raise TypeError(
                'modelscope error: Only support "str" or "tuple (img1, img2)" , but got {type(input)}'
            )

        if input_type == 0:
            logger.info('Processing Similar Image Generation mode')
        if input_type == 1:
            logger.info('Processing Interpolation mode')

        img_list = []
        for i, input_img in enumerate(input_img_list):
            img = LoadImage.convert_to_img(input_img)
            logger.info(f'Load {i}-th image done')
            img_list.append(img)

        transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(
                self.cfg.DATA.scale_size,
                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.DATA.mean, std=self.cfg.DATA.std)
        ])

        y_list = []
        for img in img_list:
            img = transforms(img)
            imgs = torch.unsqueeze(img, 0)
            imgs = imgs.to(self._device)
            imgs_x0 = self.autoencoder.encode(imgs)
            b, c, h, w = imgs_x0.shape
            aug_imgs = TF.normalize(
                F.interpolate(
                    imgs.add(1).div(2), (self.cfg.Params.vit.vit_image_size,
                                         self.cfg.Params.vit.vit_image_size),
                    mode='bilinear',
                    align_corners=True), self.cfg.Params.vit.vit_mean,
                self.cfg.Params.vit.vit_std)
            uy = self.vit(aug_imgs)
            y = F.normalize(uy, p=2, dim=1)
            y_list.append(y)

        if input_type == 0:
            result = {
                'image_data': y_list[0],
                'c': c,
                'h': h,
                'w': w,
                'type': input_type
            }
        elif input_type == 1:
            result = {
                'image_data': y_list[0],
                'image_data_s': y_list[1],
                'c': c,
                'h': h,
                'w': w,
                'type': input_type
            }
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        type_ = input['type']
        if type_ == 0:
            # Similar Image Generation #
            y = input['image_data']

            # fix seed
            torch.manual_seed(1 * 8888)
            torch.cuda.manual_seed(1 * 8888)
            i_y = y.repeat(self.repetition, 1)

            # sample images
            x0 = self.diffusion.ddim_sample_loop(
                noise=torch.randn(self.repetition, input['c'], input['h'],
                                  input['w']).to(self._device),
                model=self.decoder,
                model_kwargs=[{
                    'y': i_y
                }, {
                    'y': torch.zeros_like(i_y)
                }],
                guide_scale=1.0,
                clamp=None,
                ddim_timesteps=50,
                eta=1.0)
            i_gen_imgs = self.autoencoder.decode(x0)
            return {OutputKeys.OUTPUT_IMG: i_gen_imgs}
        else:
            # Interpolation #
            # get content-style pairs
            y = input['image_data']
            y_s = input['image_data_s']

            # fix seed
            torch.manual_seed(1 * 8888)
            torch.cuda.manual_seed(1 * 8888)
            noise = torch.randn(self.repetition, input['c'], input['h'],
                                input['w']).to(self._device)

            # interpolation between y_cid and y_sid
            factors = torch.linspace(0, 1, self.repetition).unsqueeze(1).to(
                self._device)
            i_y = (1 - factors) * y + factors * y_s

            # sample images
            x0 = self.diffusion.ddim_sample_loop(
                noise=noise,
                model=self.decoder,
                model_kwargs=[{
                    'y': i_y
                }, {
                    'y': torch.zeros_like(i_y)
                }],
                guide_scale=3.0,
                clamp=None,
                ddim_timesteps=50,
                eta=0.0)
            i_gen_imgs = self.autoencoder.decode(x0)
            return {OutputKeys.OUTPUT_IMG: i_gen_imgs}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
