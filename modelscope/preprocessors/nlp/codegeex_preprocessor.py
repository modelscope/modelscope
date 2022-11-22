# Copyright (c) 2022 Zhipu.AI

import re
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from modelscope.metainfo import Models, Preprocessors
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, InputFields, ModeKeys, ModelFile
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(Fields.nlp, module_name=Preprocessors.codegeex)
class CodeGeeXPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        """preprocess the data
        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

    @type_assert(object, (str, tuple, Dict))
    def __call__(self, data: Union[str, tuple, Dict]) -> Dict[str, Any]:
        return data
