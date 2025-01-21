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
    CardDetectionCorrectionModel
from modelscope.pipelines.cv.ocr_utils.table_process import (
    bbox_decode, bbox_post_process, decode_by_ind, get_affine_transform, nms)
from modelscope.preprocessors import load_image
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import (create_device, device_placement,
                                     verify_device)
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.card_detection_correction,
    module_name=Pipelines.card_detection_correction)
class CardDetectionCorrection(Pipeline):
    r""" Card Detection Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> detector = pipeline(Tasks.card_detection_correction, model='damo/cv_resnet18_card_correction')
    >>> detector("https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/card_detection_correction.jpg")
    >>>   {
    >>>    "polygons": array([[ 60.562023, 110.682144, 688.57715, 77.34028, 720.2409,
    >>>                         480.33508,  70.20054, 504.9171 ]], dtype=float32),
    >>>    "output_imgs": [array([
    >>>    [[[168, 176, 192],
    >>>     [165, 173, 188],
    >>>     [163, 172, 187],
    >>>     ...,
    >>>     [153, 153, 165],
    >>>     [153, 153, 165],
    >>>     [153, 153, 165]],
    >>>    [[187, 194, 210],
    >>>     [184, 192, 203],
    >>>     [183, 191, 199],
    >>>     ...,
    >>>     [168, 166, 186],
    >>>     [169, 166, 185],
    >>>     [169, 165, 184]],
    >>>    [[186, 193, 211],
    >>>     [183, 191, 205],
    >>>     [183, 192, 203],
    >>>     ...,
    >>>     [170, 167, 187],
    >>>     [171, 165, 186],
    >>>     [170, 164, 184]]]], dtype=uint8)}
    """

    def __init__(self,
                 model: str,
                 device: str = 'gpu',
                 device_map=None,
                 **kwargs):
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

        if device_map is not None:
            assert device == 'gpu', '`device` and `device_map` cannot be input at the same time!'
        self.device_map = device_map
        verify_device(device)
        self.device_name = device
        self.device = create_device(self.device_name)

        self.infer_model = CardDetectionCorrectionModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            self.infer_model.load_state_dict(checkpoint['state_dict'])
        else:
            self.infer_model.load_state_dict(checkpoint)
        self.infer_model = self.infer_model.to(self.device)
        self.infer_model.to(self.device).eval()

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)[:, :, ::-1]
        self.image = np.array(img)

        mean = np.array([0.408, 0.447, 0.470],
                        dtype=np.float32).reshape(1, 1, 3)
        std = np.array([0.289, 0.274, 0.278],
                       dtype=np.float32).reshape(1, 1, 3)
        height, width = img.shape[0:2]
        inp_height, inp_width = self.cfg.input_h, self.cfg.input_w
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

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def crop_image(self, img, position):
        x0, y0 = position[0][0], position[0][1]
        x1, y1 = position[1][0], position[1][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]

        img_width = self.distance((x0 + x3) / 2, (y0 + y3) / 2, (x1 + x2) / 2,
                                  (y1 + y2) / 2)
        img_height = self.distance((x0 + x1) / 2, (y0 + y1) / 2, (x2 + x3) / 2,
                                   (y2 + y3) / 2)

        corners_trans = np.zeros((4, 2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width, 0]
        corners_trans[2] = [img_width, img_height]
        corners_trans[3] = [0, img_height]

        transform = cv2.getPerspectiveTransform(position, corners_trans)
        dst = cv2.warpPerspective(img, transform,
                                  (int(img_width), int(img_height)))
        return dst

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred = self.infer_model(input['img'])
        return {'results': pred, 'meta': input['meta']}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        output = inputs['results'][0]
        meta = inputs['meta']
        hm = output['hm'].sigmoid_()
        wh = output['wh']
        reg = output['reg']
        angle_cls = output['cls'].sigmoid_()
        ftype_cls = output['ftype'].sigmoid_()

        bbox, inds = bbox_decode(hm, wh, reg=reg, K=self.K)
        angle_cls = decode_by_ind(
            angle_cls, inds, K=self.K).detach().cpu().numpy()
        ftype_cls = decode_by_ind(
            ftype_cls, inds,
            K=self.K).detach().cpu().numpy().astype(np.float32)
        bbox = bbox.detach().cpu().numpy()
        for i in range(bbox.shape[1]):
            bbox[0][i][9] = angle_cls[0][i]
        bbox = np.concatenate((bbox, np.expand_dims(ftype_cls, axis=-1)),
                              axis=-1)
        bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [meta['c'].cpu().numpy()],
                                 [meta['s']], meta['out_height'],
                                 meta['out_width'])

        res = []
        angle = []
        sub_imgs = []
        ftype = []
        score = []
        for idx, box in enumerate(bbox[0]):
            if box[8] > 0.3:
                angle.append(int(box[9]))
                res.append(box[0:8])
                sub_img = self.crop_image(self.image,
                                          res[-1].copy().reshape(4, 2))
                if angle[-1] == 1:
                    sub_img = cv2.rotate(sub_img, 2)
                if angle[-1] == 2:
                    sub_img = cv2.rotate(sub_img, 1)
                if angle[-1] == 3:
                    sub_img = cv2.rotate(sub_img, 0)
                sub_imgs.append(sub_img)
                ftype.append(int(box[10]))
                score.append(box[8])

        result = {
            OutputKeys.POLYGONS: res,
            OutputKeys.SCORES: score,
            OutputKeys.OUTPUT_IMGS: sub_imgs,
            OutputKeys.LABELS: angle,
            OutputKeys.LAYOUT: np.array(ftype)
        }
        return result
