# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict

from ...metainfo import Preprocessors
from ...utils.config import Config
from ...utils.constant import Fields, ModelFile
from ...utils.type_assert import type_assert
from ..base import Preprocessor
from ..builder import PREPROCESSORS
from .fields.gen_field import MultiWOZBPETextField

__all__ = ['DialogModelingPreprocessor']


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.dialog_modeling_preprocessor)
class DialogModelingPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, ModelFile.CONFIGURATION))

        import torch
        self.config.use_gpu = self.config.use_gpu and torch.cuda.is_available()

        self.text_field = MultiWOZBPETextField(
            self.model_dir, config=self.config)

    @type_assert(object, Dict)
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        import torch
        first_turn = True if len(data['history']) == 0 else False
        user_ids = self.text_field.get_ids(data['user_input'])
        inputs, prompt_id = self.text_field.convert_turn_eval(
            turn={'user': user_ids},
            pv_turn=data['history'],
            first_turn=first_turn)
        batch, batch_size = self.text_field.collate_fn_multi_turn(
            samples=[inputs])

        data['first_turn'] = first_turn
        data['batch'] = batch
        data['batch_size'] = batch_size
        data['prompt_id'] = prompt_id
        data['labels'] = [
            torch.Tensor(item).int() for item in inputs['labels']
        ]

        return data
