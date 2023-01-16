# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.util import batch_process
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Fields, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.image_classification, module_name=Pipelines.image_classification)
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
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.easyrobust_classification)
@PIPELINES.register_module(
    Tasks.image_classification,
    module_name=Pipelines.bnext_small_image_classification)
class GeneralImageClassificationPipeline(Pipeline):

    def __init__(self,
                 model: str,
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """Use `model` and `preprocessor` to create an image classification pipeline for prediction
        Args:
            model: A str format model id or model local dir to build the model instance from.
            preprocessor: A preprocessor instance to preprocess the data, if None,
            the pipeline will try to build the preprocessor according to the configuration.json file.
            kwargs: The args needed by the `Pipeline` class.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)
        self.target_gpus = None
        if preprocessor is None:
            assert hasattr(self.model, 'model_dir'), 'Model used in ImageClassificationPipeline should has ' \
                                                     'a `model_dir` attribute to build a preprocessor.'
            if self.model.__class__.__name__ == 'OfaForAllTasks':
                self.preprocessor = Preprocessor.from_pretrained(
                    model_name_or_path=self.model.model_dir,
                    type=Preprocessors.ofa_tasks_preprocessor,
                    field=Fields.multi_modal,
                    **kwargs)
            else:
                if next(self.model.parameters()).is_cuda:
                    self.target_gpus = [next(self.model.parameters()).device]
                assert hasattr(self.model, 'model_dir'), 'Model used in GeneralImageClassificationPipeline' \
                                                         ' should has a `model_dir` attribute to build a preprocessor.'
                self.preprocessor = Preprocessor.from_pretrained(
                    self.model.model_dir, **kwargs)
                if self.preprocessor.__class__.__name__ == 'ImageClassificationBypassPreprocessor':
                    from modelscope.preprocessors import ImageClassificationMmcvPreprocessor
                    self.preprocessor = ImageClassificationMmcvPreprocessor(
                        self.model.model_dir, **kwargs)
        logger.info('load model done')

    def _batch(self, data):
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return batch_process(self.model, data)
        else:
            return super()._batch(data)

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        if self.model.__class__.__name__ == 'OfaForAllTasks':
            return super().preprocess(input, **preprocess_params)
        else:
            img = LoadImage.convert_to_ndarray(input)
            img = img[:, :, ::-1]  # Convert to BGR
            data = super().preprocess(img, **preprocess_params)
            from mmcv.parallel import collate, scatter
            data = collate([data], samples_per_gpu=1)
            if self.target_gpus is not None:
                # scatter to specified GPU
                data = scatter(data, self.target_gpus)[0]
            return data

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        if self.model.__class__.__name__ != 'OfaForAllTasks':
            input['return_loss'] = False
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:

        if self.model.__class__.__name__ != 'OfaForAllTasks':
            scores = inputs

            pred_scores = np.sort(scores, axis=1)[0][::-1][:5]
            pred_labels = np.argsort(scores, axis=1)[0][::-1][:5]

            result = {
                'pred_score': [score for score in pred_scores],
                'pred_class':
                [self.model.CLASSES[label] for label in pred_labels]
            }

            outputs = {
                OutputKeys.SCORES: result['pred_score'],
                OutputKeys.LABELS: result['pred_class']
            }
            return outputs
        else:
            return inputs
