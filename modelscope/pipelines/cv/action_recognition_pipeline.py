# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.action_recognition import (BaseVideoModel,
                                                     PatchShiftTransformer)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import ReadVideoData
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.action_recognition, module_name=Pipelines.action_recognition)
class ActionRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a action recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)

        self.infer_model = BaseVideoModel(cfg=self.cfg).to(self.device)
        self.infer_model.eval()
        self.infer_model.load_state_dict(
            torch.load(model_path, map_location=self.device)['model_state'])
        self.label_mapping = self.cfg.label_mapping
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            video_input_data = ReadVideoData(self.cfg, input).to(self.device)
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        result = {'video_data': video_input_data}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred = self.perform_inference(input['video_data'])
        output_label = self.label_mapping[str(pred)]
        return {OutputKeys.LABELS: output_label}

    @torch.no_grad()
    def perform_inference(self, data, max_bsz=4):
        iter_num = math.ceil(data.size(0) / max_bsz)
        preds_list = []
        for i in range(iter_num):
            preds_list.append(
                self.infer_model(data[i * max_bsz:(i + 1) * max_bsz])[0])
        pred = torch.cat(preds_list, dim=0)
        return pred.mean(dim=0).argmax().item()

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


@PIPELINES.register_module(
    Tasks.action_recognition, module_name=Pipelines.pst_action_recognition)
class PSTActionRecognitionPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a PST action recognition pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        self.infer_model = PatchShiftTransformer(model).to(self.device)
        self.infer_model.eval()
        self.infer_model.load_state_dict(
            torch.load(model_path, map_location=self.device)['state_dict'])
        self.label_mapping = self.cfg.label_mapping
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            video_input_data = ReadVideoData(self.cfg, input).to(self.device)
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        result = {'video_data': video_input_data}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred = self.perform_inference(input['video_data'])
        output_label = self.label_mapping[str(pred)]
        return {OutputKeys.LABELS: output_label}

    @torch.no_grad()
    def perform_inference(self, data, max_bsz=4):
        iter_num = math.ceil(data.size(0) / max_bsz)
        preds_list = []
        for i in range(iter_num):
            preds_list.append(
                self.infer_model(data[i * max_bsz:(i + 1) * max_bsz]))
        pred = torch.cat(preds_list, dim=0)
        return pred.mean(dim=0).argmax().item()

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
