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
from modelscope.pipelines.cv.ocr_utils.model_convnext_transformer import \
    OCRRecModel
from modelscope.preprocessors import load_image
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

# constant
NUM_CLASSES = 7644
IMG_HEIGHT = 32
IMG_WIDTH = 300
PRED_LENTH = 75
PRED_PAD = 6


@PIPELINES.register_module(
    Tasks.ocr_recognition, module_name=Pipelines.ocr_recognition)
class OCRRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        label_path = osp.join(self.model, 'label_dict.txt')
        logger.info(f'loading model from {model_path}')

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = OCRRecModel(NUM_CLASSES).to(self.device)
        self.infer_model.eval()
        self.infer_model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        self.labelMapping = dict()
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            cnt = 2
            for line in lines:
                line = line.strip('\n')
                self.labelMapping[cnt] = line
                cnt += 1

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            img = np.array(load_image(input).convert('L'))
        elif isinstance(input, PIL.Image.Image):
            img = np.array(input.convert('L'))
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 3:
                img = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
        else:
            raise TypeError(f'input should be either str, PIL.Image,'
                            f' np.array, but got {type(input)}')
        data = []
        img_h, img_w = img.shape
        wh_ratio = img_w / img_h
        true_w = int(IMG_HEIGHT * wh_ratio)
        split_batch_cnt = 1
        if true_w < IMG_WIDTH * 1.2:
            img = cv2.resize(img, (min(true_w, IMG_WIDTH), IMG_HEIGHT))
        else:
            split_batch_cnt = math.ceil((true_w - 48) * 1.0 / 252)
            img = cv2.resize(img, (true_w, IMG_HEIGHT))

        if split_batch_cnt == 1:
            mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
            mask[:, :img.shape[1]] = img
            data.append(mask)
        else:
            for idx in range(split_batch_cnt):
                mask = np.zeros((IMG_HEIGHT, IMG_WIDTH))
                left = (PRED_LENTH * 4 - PRED_PAD * 4) * idx
                trunk_img = img[:, left:min(left + PRED_LENTH * 4, true_w)]
                mask[:, :trunk_img.shape[1]] = trunk_img
                data.append(mask)

        data = torch.FloatTensor(data).view(
            len(data), 1, IMG_HEIGHT, IMG_WIDTH) / 255.
        data = data.to(self.device)

        result = {'img': data}

        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred = self.infer_model(input['img'])
        return {'results': pred}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        preds = inputs['results']
        batchSize, length = preds.shape
        pred_idx = []
        if batchSize == 1:
            pred_idx = preds[0].cpu().data.tolist()
        else:
            for idx in range(batchSize):
                if idx == 0:
                    pred_idx.extend(preds[idx].cpu().data[:PRED_LENTH
                                                          - PRED_PAD].tolist())
                elif idx == batchSize - 1:
                    pred_idx.extend(preds[idx].cpu().data[PRED_PAD:].tolist())
                else:
                    pred_idx.extend(preds[idx].cpu().data[PRED_PAD:PRED_LENTH
                                                          - PRED_PAD].tolist())

        # ctc decoder
        last_p = 0
        str_pred = []
        for p in pred_idx:
            if p != last_p and p != 0:
                str_pred.append(self.labelMapping[p])
            last_p = p

        final_str = ''.join(str_pred)
        result = {OutputKeys.TEXT: final_str}
        return result
