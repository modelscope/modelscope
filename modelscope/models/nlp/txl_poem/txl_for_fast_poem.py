# Copyright (c) 2022 Zhipu.AI

import os
from typing import Dict

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from .fastpoem import fast_poem, prepare_model


@MODELS.register_module(Tasks.fast_poem, module_name=Models.txl)
class TXLForFastPoem(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the fast poem model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        # initialize model
        self.model, self.tokenizer, self.args = prepare_model(model_dir)

    def forward(self, input: Dict[str, str]) -> Dict[str, str]:
        pass

    def generate(self, input: Dict[str, str]) -> Dict[str, str]:
        res = fast_poem(input, self.model, self.tokenizer, self.args)
        return {OutputKeys.TEXT: res['text']}
