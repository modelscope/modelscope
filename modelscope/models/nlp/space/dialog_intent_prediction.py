# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.nlp.space import SpaceGenerator, SpaceModelBase
from modelscope.preprocessors.nlp import IntentBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['SpaceForDialogIntent']


@MODELS.register_module(
    Tasks.task_oriented_conversation, module_name=Models.space_intent)
class SpaceForDialogIntent(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            text_field (`BPETextField`, *optional*, defaults to `IntentBPETextField`):
                The text field.
            config (`Config`, *optional*, defaults to config in model hub):
                The config.
        """

        super().__init__(model_dir, *args, **kwargs)
        from modelscope.trainers.nlp.space.trainer.intent_trainer import \
            IntentTrainer
        self.model_dir = model_dir
        self.config = kwargs.pop(
            'config',
            Config.from_file(
                os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        self.text_field = kwargs.pop(
            'text_field',
            IntentBPETextField(self.model_dir, config=self.config))

        self.generator = SpaceGenerator.create(
            self.config, reader=self.text_field)
        self.model = SpaceModelBase.create(
            model_dir=model_dir,
            config=self.config,
            reader=self.text_field,
            generator=self.generator)

        def to_tensor(array):
            """
            numpy array -> tensor
            """
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.config.use_gpu else array

        self.trainer = IntentTrainer(
            model=self.model,
            to_tensor=to_tensor,
            config=self.config,
            reader=self.text_field)
        self.trainer.load()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'pred': array([2.62349960e-03 4.12110658e-03 4.12748595e-05 3.77560973e-05
                                1.08599677e-04 1.72710388e-05 2.95618793e-05 1.93638436e-04
                                6.45841064e-05 1.15997791e-04 5.11605394e-05 9.87020373e-01
                                2.66957268e-05 4.72324500e-05 9.74208378e-05], dtype=float32),
                    }
        Example:
            >>> from modelscope.hub.snapshot_download import snapshot_download
            >>> from modelscope.models.nlp import SpaceForDialogIntent
            >>> from modelscope.preprocessors import DialogIntentPredictionPreprocessor
            >>> cache_path = snapshot_download('damo/nlp_space_dialog-intent-prediction')
            >>> preprocessor = DialogIntentPredictionPreprocessor(model_dir=cache_path)
            >>> model = SpaceForDialogIntent(
                    model_dir=cache_path,
                    text_field=preprocessor.text_field,
                    config=preprocessor.config)
            >>> print(model(preprocessor("What do I need to do for the card activation?")))
        """
        import numpy as np
        pred = self.trainer.forward(input)
        pred = np.squeeze(pred[0], 0)

        return {'pred': pred}
