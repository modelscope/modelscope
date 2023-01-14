# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import (MPlugPreprocessor, OfaPreprocessor,
                                      Preprocessor)
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_captioning, module_name=Pipelines.image_captioning)
class ImageCaptioningPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        """
        use `model` and `preprocessor` to create a image captioning pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        if preprocessor is None:
            if isinstance(self.model, OfaForAllTasks):
                self.preprocessor = OfaPreprocessor(self.model.model_dir)
            elif isinstance(self.model, MPlugForAllTasks):
                self.preprocessor = MPlugPreprocessor(self.model.model_dir)

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        elif isinstance(self.model, MPlugForAllTasks):
            from transformers.tokenization_utils_base import BatchEncoding
            batch_data = dict(train=data[0]['train'])
            batch_data['image'] = torch.cat([d['image'] for d in data])
            question = {}
            for k in data[0]['question'].keys():
                question[k] = torch.cat([d['question'][k] for d in data])
            batch_data['question'] = BatchEncoding(question)
            return batch_data
        else:
            return super(ImageCaptioningPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
