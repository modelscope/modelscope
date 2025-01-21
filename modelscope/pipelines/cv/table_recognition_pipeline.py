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
from modelscope.pipelines.cv.ocr_utils.model_dla34 import TableRecModel
from modelscope.pipelines.cv.ocr_utils.table_process import (
    bbox_decode, bbox_post_process, gbox_decode, gbox_post_process,
    get_affine_transform, group_bbox_by_gbox, nms)
from modelscope.preprocessors import load_image
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.table_recognition, module_name=Pipelines.table_recognition)
class TableRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')

        self.K = 1000
        self.MK = 4000
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = TableRecModel().to(self.device)
        self.infer_model.eval()
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.infer_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.infer_model.load_state_dict(checkpoint)

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)[:, :, ::-1]

        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = 1024, 1024
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
        v2c = output['v2c']
        c2v = output['c2v']
        reg = output['reg']
        bbox, _ = bbox_decode(hm[:, 0:1, :, :], c2v, reg=reg, K=self.K)
        gbox, _ = gbox_decode(hm[:, 1:2, :, :], v2c, reg=reg, K=self.MK)

        bbox = bbox.detach().cpu().numpy()
        gbox = gbox.detach().cpu().numpy()
        bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [meta['c'].cpu().numpy()],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])
        gbox = gbox_post_process(gbox.copy(), [meta['c'].cpu().numpy()],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])
        bbox = group_bbox_by_gbox(bbox[0], gbox[0])

        res = []
        for box in bbox:
            if box[8] > 0.3:
                res.append(box[0:8])

        result = {OutputKeys.POLYGONS: np.array(res)}
        return result
