# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.cv.ocr_utils.model_resnet18_half import \
    LicensePlateDet
from modelscope.pipelines.cv.ocr_utils.table_process import (
    bbox_decode, bbox_post_process, decode_by_ind, get_affine_transform, nms)
from modelscope.preprocessors import load_image
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.license_plate_detection,
    module_name=Pipelines.license_plate_detection)
class LicensePlateDetection(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading model from {model_path}')

        self.cfg = Config.from_file(config_path)
        self.K = self.cfg.K
        self.car_type = self.cfg.Type
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = LicensePlateDet()
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.infer_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.infer_model.load_state_dict(checkpoint)
        self.infer_model = self.infer_model.to(self.device)
        self.infer_model.to(self.device).eval()

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)[:, :, ::-1]

        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = 512, 512
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        resized_image = cv2.resize(img, (width, height))
        inp_image = cv2.warpAffine(
            resized_image,
            trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height,
                                                      inp_width)
        images = torch.from_numpy(images).to(self.device)
        meta = {
            'c': c,
            's': s,
            'input_height': inp_height,
            'input_width': inp_width,
            'out_height': inp_height // 4,
            'out_width': inp_width // 4
        }

        result = {'img': images, 'meta': meta}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred = self.infer_model(input['img'])
        return {'results': pred, 'meta': input['meta']}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output = inputs['results'][0]
        meta = inputs['meta']
        hm = output['hm'].sigmoid_()
        ftype = output['ftype'].sigmoid_()
        wh = output['wh']
        reg = output['reg']

        bbox, inds = bbox_decode(hm, wh, reg=reg, K=self.K)
        car_type = decode_by_ind(ftype, inds, K=self.K).detach().cpu().numpy()
        bbox = bbox.detach().cpu().numpy()
        for i in range(bbox.shape[1]):
            bbox[0][i][9] = car_type[0][i]
        bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [meta['c'].cpu().numpy()],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])

        res, Type = [], []
        for box in bbox[0]:
            if box[8] > 0.3:
                res.append(box[0:8])
                Type.append(self.car_type[int(box[9])])

        result = {OutputKeys.POLYGONS: np.array(res), OutputKeys.TEXT: Type}
        return result
