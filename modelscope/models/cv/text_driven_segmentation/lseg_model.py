# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.text_driven_segmentation import \
    TextDrivenSegmentation
from modelscope.outputs import OutputKeys
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()
__all__ = ['TextDrivenSeg']


@MODELS.register_module(
    Tasks.text_driven_segmentation,
    module_name=Models.text_driven_segmentation)
class TextDrivenSeg(TorchModel):
    """ text driven segmentation model.
    """

    def __init__(self, model_dir, device_id=0, *args, **kwargs):
        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)
        self.model = TextDrivenSegmentation(model_dir=model_dir)
        pretrained_params = torch.load('{}/{}'.format(
            model_dir, ModelFile.TORCH_MODEL_BIN_FILE))
        self.model.load_state_dict(pretrained_params)
        self.model.eval()
        if device_id >= 0 and torch.cuda.is_available():
            self.model.to('cuda:{}'.format(device_id))
            logger.info('Use GPU: {}'.format(device_id))
        else:
            device_id = -1
            logger.info('Use CPU for inference')
        self.device_id = device_id

    def preprocess(self, img, size=640):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        h, w, c = img.shape
        max_hw = max(h, w)
        ratio = 1.0 * size / max_hw
        crop_h, crop_w = int(ratio * h), int(ratio * w)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((crop_w, crop_h), Image.BILINEAR)
        np_img = np.array(pil_img, dtype=np.float32) / 255.
        for j in range(3):
            np_img[:, :, j] = (np_img[:, :, j] - mean[j]) / std[j]
        img_pad = np.zeros((size, size, 3), dtype=np.float32)
        img_pad[:crop_h, :crop_w] = np_img
        img_pad = torch.from_numpy(img_pad).permute(2, 0,
                                                    1).unsqueeze(0).float()
        return img_pad, h, w, crop_h, crop_w

    def postprocess(self, tensors, crop_h, crop_w, ori_h, ori_w):
        output = np.clip(tensors * 255., a_min=0, a_max=255.)
        crop_output = np.array(output[:crop_h, :crop_w], dtype=np.uint8)
        pil_output = Image.fromarray(crop_output)
        pil_output = pil_output.resize((ori_w, ori_h), Image.BILINEAR)
        np_output = np.array(pil_output, dtype=np.uint8)
        np_output[np_output < 128] = 0
        np_output[np_output >= 128] = 255
        np_output = np.uint8(np_output)
        return np_output

    def forward(self, image, text):
        """
        image should be numpy array, dtype=np.uint8, shape: height*width*3
        """
        image_tensor, ori_h, ori_w, crop_h, crop_w = self.preprocess(
            image, size=640)
        pred = self.inference(image_tensor, text)
        msk = self.postprocess(pred, crop_h, crop_w, ori_h, ori_w, size=640)
        outputs = {OutputKeys.MASKS: msk}
        return outputs

    def inference(self, image, text):
        """
        image should be tensor, 1 * 3 * 640 * 640
        """
        with torch.no_grad():
            if self.device_id == -1:
                output = self.model(image, [text])
            else:
                device = torch.device('cuda', self.device_id)
                output = self.model(image.to(device), [text])
            output = F.interpolate(output, size=(640, 640), mode='bilinear')
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            output = output[0]
            if self.device_id == -1:
                pred = output.data.numpy()
            else:
                pred = output.data.cpu().numpy()
            del output
        return pred
