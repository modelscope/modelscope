# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from typing import Any, Dict

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models.cv.crowd_counting import HRNetCrowdCounting
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.crowd_counting, module_name=Pipelines.crowd_counting)
class CrowdCountingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        assert isinstance(model, str), 'model must be a single str'
        super().__init__(model=model, auto_collate=False, **kwargs)
        logger.info(f'loading model from dir {model}')
        self.infer_model = HRNetCrowdCounting(model).to(self.device)
        self.infer_model.eval()
        logger.info('load model done')

    def resize(self, img):
        height = img.size[1]
        width = img.size[0]
        resize_height = height
        resize_width = width
        if resize_width >= 2048:
            tmp = resize_width
            resize_width = 2048
            resize_height = (resize_width / tmp) * resize_height

        if resize_height >= 2048:
            tmp = resize_height
            resize_height = 2048
            resize_width = (resize_height / tmp) * resize_width

        if resize_height <= 416:
            tmp = resize_height
            resize_height = 416
            resize_width = (resize_height / tmp) * resize_width
        if resize_width <= 416:
            tmp = resize_width
            resize_width = 416
            resize_height = (resize_width / tmp) * resize_height

        # other constraints
        if resize_height < resize_width:
            if resize_width / resize_height > 2048 / 416:  # 1024/416=2.46
                resize_width = 2048
                resize_height = 416
        else:
            if resize_height / resize_width > 2048 / 416:
                resize_height = 2048
                resize_width = 416

        resize_height = math.ceil(resize_height / 32) * 32
        resize_width = math.ceil(resize_width / 32) * 32
        img = transforms.Resize([resize_height, resize_width])(img)
        return img

    def merge_crops(self, eval_shape, eval_p, pred_m):
        for i in range(3):
            for j in range(3):
                start_h, start_w = math.floor(eval_shape[2] / 4), math.floor(
                    eval_shape[3] / 4)
                valid_h, valid_w = eval_shape[2] // 2, eval_shape[3] // 2
                pred_h = math.floor(
                    3 * eval_shape[2] / 4) + (eval_shape[2] // 2) * (
                        i - 1)
                pred_w = math.floor(
                    3 * eval_shape[3] / 4) + (eval_shape[3] // 2) * (
                        j - 1)
                if i == 0:
                    valid_h = math.floor(3 * eval_shape[2] / 4)
                    start_h = 0
                    pred_h = 0
                elif i == 2:
                    valid_h = math.ceil(3 * eval_shape[2] / 4)

                if j == 0:
                    valid_w = math.floor(3 * eval_shape[3] / 4)
                    start_w = 0
                    pred_w = 0
                elif j == 2:
                    valid_w = math.ceil(3 * eval_shape[3] / 4)
                pred_m[:, :, pred_h:pred_h + valid_h, pred_w:pred_w
                       + valid_w] += eval_p[i * 3 + j:i * 3 + j + 1, :,
                                            start_h:start_h + valid_h,
                                            start_w:start_w + valid_w]
        return pred_m

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_img(input)
        img = self.resize(img)
        img_ori_tensor = transforms.ToTensor()(img)
        img_shape = img_ori_tensor.shape
        img = transforms.Normalize((0.485, 0.456, 0.406),
                                   (0.229, 0.224, 0.225))(
                                       img_ori_tensor)
        patch_height, patch_width = (img_shape[1]) // 2, (img_shape[2]) // 2
        imgs = []
        for i in range(3):
            for j in range(3):
                start_h, start_w = (patch_height // 2) * i, (patch_width
                                                             // 2) * j
                imgs.append(img[:, start_h:start_h + patch_height,
                                start_w:start_w + patch_width])

        imgs = torch.stack(imgs)
        eval_img = imgs.to(self.device)
        eval_patchs = torch.squeeze(eval_img)
        prediction_map = torch.zeros(
            (1, 1, img_shape[1] // 2, img_shape[2] // 2)).to(self.device)
        result = {
            'img': eval_patchs,
            'map': prediction_map,
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        counts, img_data = self.perform_inference(input)
        return {OutputKeys.SCORES: counts, OutputKeys.OUTPUT_IMG: img_data}

    @torch.no_grad()
    def perform_inference(self, data):
        eval_patchs = data['img']
        prediction_map = data['map']
        eval_prediction, _, _ = self.infer_model(eval_patchs)
        eval_patchs_shape = eval_prediction.shape
        prediction_map = self.merge_crops(eval_patchs_shape, eval_prediction,
                                          prediction_map)

        return torch.sum(
            prediction_map, dim=(
                1, 2,
                3)).data.cpu().numpy(), prediction_map.data.cpu().numpy()[0][0]

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
