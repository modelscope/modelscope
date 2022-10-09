# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os.path as osp
import sys
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image

import modelscope.models.cv.image_to_image_translation.data as data
import modelscope.models.cv.image_to_image_translation.models as models
import modelscope.models.cv.image_to_image_translation.ops as ops
from modelscope.fileio import File
from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_to_image_translation.model_translation import \
    UNet
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


def save_grid(imgs, filename, nrow=5):
    save_image(
        imgs.clamp(-1, 1), filename, range=(-1, 1), normalize=True, nrow=nrow)


@PIPELINES.register_module(
    Tasks.image_to_image_translation,
    module_name=Pipelines.image2image_translation)
class Image2ImageTranslationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a kws pipeline for prediction
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

        # load palette model
        palette_model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading palette model from {palette_model_path}')
        self.palette = UNet(
            resolution=self.cfg.Params.unet.unet_resolution,
            in_dim=self.cfg.Params.unet.unet_in_dim,
            dim=self.cfg.Params.unet.unet_dim,
            context_dim=self.cfg.Params.unet.unet_context_dim,
            out_dim=self.cfg.Params.unet.unet_out_dim,
            dim_mult=self.cfg.Params.unet.unet_dim_mult,
            num_heads=self.cfg.Params.unet.unet_num_heads,
            head_dim=None,
            num_res_blocks=self.cfg.Params.unet.unet_res_blocks,
            attn_scales=self.cfg.Params.unet.unet_attn_scales,
            num_classes=self.cfg.Params.unet.unet_num_classes + 1,
            dropout=self.cfg.Params.unet.unet_dropout).eval().requires_grad_(
                False).to(self._device)
        self.palette.load_state_dict(
            torch.load(palette_model_path, map_location=self._device))
        logger.info('load palette model done')

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

        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(
                self.cfg.DATA.scale_size,
                interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.cfg.DATA.mean, std=self.cfg.DATA.std)
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if len(input) == 3:  # colorization
            _, input_type, save_path = input
        elif len(input) == 4:  # uncropping or in-painting
            _, meta, input_type, save_path = input
            if input_type == 0:  # uncropping
                assert meta in ['up', 'down', 'left', 'right']
                direction = meta

        list_ = []
        for i in range(len(input) - 2):
            input_img = input[i]
            if input_img in ['up', 'down', 'left', 'right']:
                continue
            if isinstance(input_img, str):
                if input_type == 2 and i == 0:
                    logger.info('Loading image by origin way ... ')
                    bytes = File.read(input_img)
                    img = Image.open(io.BytesIO(bytes))
                    assert len(img.split()) == 4
                else:
                    img = load_image(input_img)
            elif isinstance(input_img, PIL.Image.Image):
                img = input_img.convert('RGB')
            elif isinstance(input_img, np.ndarray):
                if len(input_img.shape) == 2:
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_GRAY2BGR)
                img = input_img[:, :, ::-1]
                img = Image.fromarray(img.astype('uint8')).convert('RGB')
            else:
                raise TypeError(f'input should be either str, PIL.Image,'
                                f' np.array, but got {type(input)}')
            list_.append(img)
        img_list = []
        if input_type != 2:
            for img in list_:
                img = self.transforms(img)
                imgs = torch.unsqueeze(img, 0)
                imgs = imgs.to(self._device)
                img_list.append(imgs)
        elif input_type == 2:
            mask, masked_img = list_[0], list_[1]
            img = self.transforms(masked_img.convert('RGB'))
            mask = torch.from_numpy(
                np.array(
                    mask.resize((img.shape[2], img.shape[1])),
                    dtype=np.float32)[:, :, -1] / 255.0).unsqueeze(0)
            img = (1 - mask) * img + mask * torch.randn_like(img).clamp_(-1, 1)
            imgs = img.unsqueeze(0).to(self._device)
        b, c, h, w = imgs.shape
        y = torch.LongTensor([self.cfg.Classes.class_id]).to(self._device)

        if input_type == 0:
            assert len(img_list) == 1
            result = {
                'image_data': img_list[0],
                'c': c,
                'h': h,
                'w': w,
                'direction': direction,
                'type': input_type,
                'y': y,
                'save_path': save_path
            }
        elif input_type == 1:
            assert len(img_list) == 1
            result = {
                'image_data': img_list[0],
                'c': c,
                'h': h,
                'w': w,
                'type': input_type,
                'y': y,
                'save_path': save_path
            }
        elif input_type == 2:
            result = {
                'image_data': imgs,
                # 'image_mask': mask,
                'c': c,
                'h': h,
                'w': w,
                'type': input_type,
                'y': y,
                'save_path': save_path
            }
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        type_ = input['type']
        if type_ == 0:
            # Uncropping
            img = input['image_data']
            direction = input['direction']
            y = input['y']

            # fix seed
            torch.manual_seed(1 * 8888)
            torch.cuda.manual_seed(1 * 8888)

            logger.info(f'Processing {direction} uncropping')
            img = img.clone()
            i_y = y.repeat(self.repetition, 1)
            if direction == 'up':
                img[:, :, input['h'] // 2:, :] = torch.randn_like(
                    img[:, :, input['h'] // 2:, :])
            elif direction == 'down':
                img[:, :, :input['h'] // 2, :] = torch.randn_like(
                    img[:, :, :input['h'] // 2, :])
            elif direction == 'left':
                img[:, :, :,
                    input['w'] // 2:] = torch.randn_like(img[:, :, :,
                                                             input['w'] // 2:])
            elif direction == 'right':
                img[:, :, :, :input['w'] // 2] = torch.randn_like(
                    img[:, :, :, :input['w'] // 2])
            i_concat = self.autoencoder.encode(img).repeat(
                self.repetition, 1, 1, 1)

            # sample images
            x0 = self.diffusion.ddim_sample_loop(
                noise=torch.randn_like(i_concat),
                model=self.palette,
                model_kwargs=[{
                    'y': i_y,
                    'concat': i_concat
                }, {
                    'y':
                    torch.full_like(i_y,
                                    self.cfg.Params.unet.unet_num_classes),
                    'concat':
                    i_concat
                }],
                guide_scale=1.0,
                clamp=None,
                ddim_timesteps=50,
                eta=1.0)
            i_gen_imgs = self.autoencoder.decode(x0)
            save_grid(i_gen_imgs, input['save_path'], nrow=4)
            return {OutputKeys.OUTPUT_IMG: i_gen_imgs}

        elif type_ == 1:
            # Colorization #
            img = input['image_data']
            y = input['y']
            # fix seed
            torch.manual_seed(1 * 8888)
            torch.cuda.manual_seed(1 * 8888)

            logger.info('Processing Colorization')
            img = img.clone()
            img = img.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            i_concat = self.autoencoder.encode(img).repeat(
                self.repetition, 1, 1, 1)
            i_y = y.repeat(self.repetition, 1)

            # sample images
            x0 = self.diffusion.ddim_sample_loop(
                noise=torch.randn_like(i_concat),
                model=self.palette,
                model_kwargs=[{
                    'y': i_y,
                    'concat': i_concat
                }, {
                    'y':
                    torch.full_like(i_y,
                                    self.cfg.Params.unet.unet_num_classes),
                    'concat':
                    i_concat
                }],
                guide_scale=1.0,
                clamp=None,
                ddim_timesteps=50,
                eta=0.0)
            i_gen_imgs = self.autoencoder.decode(x0)
            save_grid(i_gen_imgs, input['save_path'], nrow=4)
            return {OutputKeys.OUTPUT_IMG: i_gen_imgs}
        elif type_ == 2:
            # Combination #
            logger.info('Processing Combination')

            # prepare inputs
            img = input['image_data']
            concat = self.autoencoder.encode(img).repeat(
                self.repetition, 1, 1, 1)
            y = torch.LongTensor([126]).unsqueeze(0).to(self._device).repeat(
                self.repetition, 1)

            # sample images
            x0 = self.diffusion.ddim_sample_loop(
                noise=torch.randn_like(concat),
                model=self.palette,
                model_kwargs=[{
                    'y': y,
                    'concat': concat
                }, {
                    'y':
                    torch.full_like(y, self.cfg.Params.unet.unet_num_classes),
                    'concat':
                    concat
                }],
                guide_scale=1.0,
                clamp=None,
                ddim_timesteps=50,
                eta=1.0)
            i_gen_imgs = self.autoencoder.decode(x0)
            save_grid(i_gen_imgs, input['save_path'], nrow=4)
            return {OutputKeys.OUTPUT_IMG: i_gen_imgs}
        else:
            raise TypeError(
                f'input type should be 0 (Uncropping), 1 (Colorization), 2 (Combation)'
                f' but got {type_}')

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
