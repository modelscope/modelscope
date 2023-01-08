# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal import OfaForAllTasks
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import OfaPreprocessor, Preprocessor, load_image
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.device import get_device
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_classification, module_name=Pipelines.image_classification)
class ImageClassificationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        assert isinstance(model, str) or isinstance(model, Model), \
            'model must be a single str or OfaForAllTasks'
        self.model.eval()
        self.model.to(get_device())
        if preprocessor is None and isinstance(self.model, OfaForAllTasks):
            self.preprocessor = OfaPreprocessor(model_dir=self.model.model_dir)

    def _batch(self, data):
        if isinstance(self.model, OfaForAllTasks):
            return batch_process(self.model, data)
        else:
            return super(ImageClassificationPipeline, self)._batch(data)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.general_image_classification)
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.daily_image_classification)
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.nextvit_small_daily_image_classification)
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.convnext_base_image_classification_garbage)
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.common_image_classification)
class GeneralImageClassificationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` and `preprocessor` to create a image classification pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        from mmcls.datasets.pipelines import Compose
        from mmcv.parallel import collate, scatter
        from modelscope.models.cv.image_classification.utils import preprocess_transform

        img = LoadImage.convert_to_ndarray(input)  # Default in RGB order
        img = img[:, :, ::-1]  # Convert to BGR

        cfg = self.model.cfg

        if self.model.config_type == 'mmcv_config':
            if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
                cfg.data.test.pipeline.pop(0)
            data = dict(img=img)
            test_pipeline = Compose(cfg.data.test.pipeline)
        else:
            if cfg.preprocessor.val[0]['type'] == 'LoadImageFromFile':
                cfg.preprocessor.val.pop(0)
            data = dict(img=img)
            data_pipeline = preprocess_transform(cfg.preprocessor.val)
            test_pipeline = Compose(data_pipeline)

        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [next(self.model.parameters()).device])[0]

        return data

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        with torch.no_grad():
            input['return_loss'] = False
            scores = self.model(input)

        return {'scores': scores}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        scores = inputs['scores']

        pred_scores = np.sort(scores, axis=1)[0][::-1][:5]
        pred_labels = np.argsort(scores, axis=1)[0][::-1][:5]

        result = {'pred_score': [score for score in pred_scores]}
        result['pred_class'] = [
            self.model.CLASSES[lable] for lable in pred_labels
        ]

        outputs = {
            OutputKeys.SCORES: result['pred_score'],
            OutputKeys.LABELS: result['pred_class']
        }
        return outputs
