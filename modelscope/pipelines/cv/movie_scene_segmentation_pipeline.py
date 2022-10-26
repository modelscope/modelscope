# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.movie_scene_segmentation,
    module_name=Pipelines.movie_scene_segmentation)
class MovieSceneSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """use `model` to create a movie scene segmentation pipeline for prediction

        Args:
            model: model id on modelscope hub
        """
        _device = kwargs.pop('device', 'gpu')
        if torch.cuda.is_available() and _device == 'gpu':
            device = 'gpu'
        else:
            device = 'cpu'
        super().__init__(model=model, device=device, **kwargs)

        logger.info('Load model done!')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        """ use pyscenedetect to detect shot from the input video, and generate key-frame jpg, anno.ndjson, and shot-frame.txt
            Then use shot-encoder to encoder feat of the detected key-frame

        Args:
            input: path of the input video

        """
        self.input_video_pth = input
        if isinstance(input, str):
            shot_feat, sid = self.model.preprocess(input)
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')

        result = {'sid': sid, 'shot_feat': shot_feat}

        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            output = self.model.inference(input)
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        data = {'input_video_pth': self.input_video_pth, 'feat': inputs}
        scene_num, scene_meta_lst, shot_num, shot_meta_lst = self.model.postprocess(
            data)
        result = {
            OutputKeys.SHOT_NUM: shot_num,
            OutputKeys.SHOT_META_LIST: shot_meta_lst,
            OutputKeys.SCENE_NUM: scene_num,
            OutputKeys.SCENE_META_LIST: scene_meta_lst
        }
        return result
