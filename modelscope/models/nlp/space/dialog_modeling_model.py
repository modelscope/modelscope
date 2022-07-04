# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Optional

from ....metainfo import Models
from ....preprocessors.space.fields.gen_field import MultiWOZBPETextField
from ....trainers.nlp.space.trainer.gen_trainer import MultiWOZTrainer
from ....utils.config import Config
from ....utils.constant import ModelFile, Tasks
from ...base import Model, Tensor
from ...builder import MODELS
from .model.generator import Generator
from .model.model_base import SpaceModelBase

__all__ = ['SpaceForDialogModeling']


@MODELS.register_module(Tasks.dialog_modeling, module_name=Models.space)
class SpaceForDialogModeling(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.config = kwargs.pop(
            'config',
            Config.from_file(
                os.path.join(self.model_dir, ModelFile.CONFIGURATION)))
        self.text_field = kwargs.pop(
            'text_field',
            MultiWOZBPETextField(self.model_dir, config=self.config))
        self.generator = Generator.create(self.config, reader=self.text_field)
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

        self.trainer = MultiWOZTrainer(
            model=self.model,
            to_tensor=to_tensor,
            config=self.config,
            reader=self.text_field,
            evaluator=None)
        self.trainer.load()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'labels': array([1,192,321,12]), # lable
                        'resp': array([293,1023,123,1123]), #vocab label for response
                        'bspn': array([123,321,2,24,1 ]),
                        'aspn': array([47,8345,32,29,1983]),
                        'db': array([19, 24, 20]),
                    }
        """

        turn = {'user': input['user']}
        old_pv_turn = input['history']

        pv_turn = self.trainer.forward(turn=turn, old_pv_turn=old_pv_turn)

        return pv_turn
