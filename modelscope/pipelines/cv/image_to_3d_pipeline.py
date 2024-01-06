# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import rembg
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from PIL import Image
from torchvision.utils import save_image

# import modelscope.models.cv.image_to_image_generation.data as data
# import modelscope.models.cv.image_to_image_generation.models as models
# import modelscope.models.cv.image_to_image_generation.ops as ops
from modelscope.metainfo import Pipelines
# from modelscope.models.cv.image_to_3d.ldm.models.diffusion.sync_dreamer import \
#     SyncMultiviewDiffusion
from modelscope.models.cv.image_to_3d.ldm.util import (add_margin,
                                                       instantiate_from_config)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

# from modelscope.models.cv.image_to_3d.model import UNet
# from modelscope.models.cv.image_to_image_generation.models.clip import \
#     VisionTransformer

logger = get_logger()


# Load Syncdreamer Model
def load_model(cfg, ckpt, strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=strict)
    model = model.cuda().eval()
    return model


# Prepare Syncdreamer Input
def prepare_inputs(image_input, elevation_input, crop_size=-1, image_size=256):
    image_input[:, :, :3] = image_input[:, :, :3][:, :, ::-1]
    image_input = Image.fromarray(image_input)
    if crop_size != -1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
        image_input = add_margin(ref_img_, size=image_size)
    else:
        image_input = add_margin(
            image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((image_size, image_size),
                                         resample=Image.BICUBIC)

    image_input = np.asarray(image_input)
    image_input = image_input.astype(np.float32) / 255.0
    ref_mask = image_input[:, :, 3:]
    image_input[:, :, :
                3] = image_input[:, :, :
                                 3] * ref_mask + 1 - ref_mask  # white background
    image_input = image_input[:, :, :3] * 2.0 - 1.0
    image_input = torch.from_numpy(image_input.astype(np.float32))
    elevation_input = torch.from_numpy(
        np.asarray([np.deg2rad(elevation_input)], np.float32))
    return {'input_image': image_input, 'input_elevation': elevation_input}


@PIPELINES.register_module(
    Tasks.image_to_3d, module_name=Pipelines.image_to_3d)
class Image23DPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a image-to-3d generation pipeline
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        # print(config_path)
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        ckpt = config_path.replace('configuration.json',
                                   'syncdreamer-pretrain.ckpt')
        self.model = load_model(
            config_path.replace('configuration.json', 'syncdreamer.yaml'),
            ckpt).to(self._device)
        # os.system("pip install -r {}".format(config_path.replace("configuration.json", "requirements.txt")))
        # assert isinstance(self.model, SyncMultiviewDiffusion)

    def preprocess(self, input: Input) -> Dict[str, Any]:

        result = rembg.remove(Image.open(input))
        print(type(result))
        img = np.array(result)
        img[:, :, :3] = img[:, :, :3][:, :, ::-1]
        # img = cv2.imread(input)
        data = prepare_inputs(
            img, elevation_input=10, crop_size=200, image_size=256)

        for k, v in data.items():
            data[k] = v.unsqueeze(0).cuda()
            data[k] = torch.repeat_interleave(
                data[k], 1, dim=0)  # only one sample
        return data

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        x_sample = self.model.sample(input, 2.0, 8)

        B, N, _, H, W = x_sample.shape
        x_sample = (torch.clamp(x_sample, max=1.0, min=-1.0) + 1) * 0.5
        x_sample = x_sample.permute(0, 1, 3, 4, 2).cpu().numpy() * 255
        x_sample = x_sample.astype(np.uint8)
        show_in_im2 = [Image.fromarray(x_sample[0, ni]) for ni in range(N)]
        return {'MViews': show_in_im2}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
