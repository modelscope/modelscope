# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import MPlugForAllTasks, OfaForAllTasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
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
        super().__init__(model=model)
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or OfaForAllTasks'
        if isinstance(model, str):
            pipe_model = Model.from_pretrained(model)
        elif isinstance(model, Model):
            pipe_model = model
        else:
            raise NotImplementedError
        pipe_model.model.eval()
        if preprocessor is None:
            if isinstance(pipe_model, OfaForAllTasks):
                preprocessor = OfaPreprocessor(pipe_model.model_dir)
            elif isinstance(pipe_model, MPlugForAllTasks):
                preprocessor = MPlugPreprocessor(pipe_model.model_dir)
        super().__init__(model=pipe_model, preprocessor=preprocessor, **kwargs)

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            # collate batch data due to the nested data structure
            if isinstance(data, list):
                batch_data = {}
                batch_data['nsentences'] = len(data)
                batch_data['samples'] = [d['samples'][0] for d in data]
                batch_data['net_input'] = {}
                for k in data[0]['net_input'].keys():
                    batch_data['net_input'][k] = torch.concat(
                        [d['net_input'][k] for d in data])

            return batch_data
        elif isinstance(self.model, MPlugForAllTasks):
            from transformers.tokenization_utils_base import BatchEncoding
            batch_data = dict(train=data[0]['train'])
            batch_data['image'] = torch.concat([d['image'] for d in data])
            question = {}
            for k in data[0]['question'].keys():
                question[k] = torch.concat([d['question'][k] for d in data])
            batch_data['question'] = BatchEncoding(question)
            return batch_data
        else:
            return super()._collate_batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
