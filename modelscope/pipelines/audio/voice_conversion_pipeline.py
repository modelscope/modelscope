# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks


@PIPELINES.register_module(
    Tasks.voice_conversion, module_name=Pipelines.voice_conversion)
class VCPipeline(Pipeline):
    r"""ANS (Acoustic Noise Suppression) Inference Pipeline .

    When invoke the class with pipeline.__call__(), it accept only one
    parameter:
        inputs(str): the path of wav file
    """
    SAMPLE_RATE = 16000

    def __init__(self, model, **kwargs):
        """
        use `model` and `preprocessor` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.model.eval()
        self.stream_mode = kwargs.get('stream_mode', False)

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            outputs = self.model(inputs)
        outputs *= 32768.
        outputs = np.array(outputs, 'int16').tobytes()
        return {OutputKeys.OUTPUT_PCM: outputs}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        return inputs
