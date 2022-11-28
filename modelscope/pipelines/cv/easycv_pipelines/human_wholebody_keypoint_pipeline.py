# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path
from typing import Any

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from .base import EasyCVPipeline


@PIPELINES.register_module(
    Tasks.human_wholebody_keypoint,
    module_name=Pipelines.human_wholebody_keypoint)
class HumanWholebodyKeypointsPipeline(EasyCVPipeline):
    """Pipeline for human wholebody 2d keypoints detection."""

    def __init__(self,
                 model: str,
                 model_file_pattern=ModelFile.TORCH_MODEL_FILE,
                 *args,
                 **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.
        """
        super(HumanWholebodyKeypointsPipeline, self).__init__(
            model=model,
            model_file_pattern=model_file_pattern,
            *args,
            **kwargs)

    def _build_predict_op(self, **kwargs):
        """Build EasyCV predictor."""
        from easycv.predictors.builder import build_predictor
        detection_predictor_type = self.cfg['DETECTION']['type']
        detection_model_path = os.path.join(
            self.model_dir, self.cfg['DETECTION']['model_path'])
        detection_cfg_file = os.path.join(self.model_dir,
                                          self.cfg['DETECTION']['config_file'])
        detection_score_threshold = self.cfg['DETECTION']['score_threshold']
        self.cfg.pipeline.predictor_config[
            'detection_predictor_config'] = dict(
                type=detection_predictor_type,
                model_path=detection_model_path,
                config_file=detection_cfg_file,
                score_threshold=detection_score_threshold)
        easycv_config = self._to_easycv_config()
        pipeline_op = build_predictor(self.cfg.pipeline.predictor_config, {
            'model_path': self.model_path,
            'config_file': easycv_config,
            **kwargs
        })
        return pipeline_op

    def __call__(self, inputs) -> Any:
        outputs = self.predict_op(inputs)

        results = [{
            OutputKeys.KEYPOINTS: output['keypoints'],
            OutputKeys.BOXES: output['boxes']
        } for output in outputs]

        if self._is_single_inputs(inputs):
            results = results[0]

        return results
