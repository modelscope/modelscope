# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks

__all__ = ['PalmForTextGeneration']


@MODELS.register_module(Tasks.text_generation, module_name=Models.palm)
class PalmForTextGeneration(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the text generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """
        super().__init__(model_dir, *args, **kwargs)

        from modelscope.models.nlp.palm_v2 import (
            PalmForConditionalGeneration, Translator)
        self.model = PalmForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = self.model.tokenizer
        self.generator = Translator(self.model)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'loss': Tensor([12.34]), # loss for backward
                    }
        """
        return self.model(**input)

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        outputs = self.generator(**input)
        preds = outputs['predictions']
        return {'sequences': [pred[0] for pred in preds]}
