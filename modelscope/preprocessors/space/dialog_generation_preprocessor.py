# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import uuid
from typing import Any, Dict, Union

from modelscope.preprocessors.space.fields.gen_field import \
    MultiWOZBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, InputFields
from modelscope.utils.type_assert import type_assert
from ..base import Preprocessor
from ..builder import PREPROCESSORS

__all__ = ['DialogGenerationPreprocessor']


@PREPROCESSORS.register_module(Fields.nlp, module_name=r'space-generation')
class DialogGenerationPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

        self.model_dir: str = model_dir
        self.config = Config.from_file(
            os.path.join(self.model_dir, 'configuration.json'))
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

        user_ids = self.text_field.get_ids(data['user_input'])
        data['user'] = user_ids

        return data
