# Copyright (c) 2022 Zhipu.AI

import os.path as osp
import re
from typing import Any, Dict, Iterable, Optional, Tuple, Union

from modelscope.metainfo import Models, Preprocessors
from modelscope.outputs import OutputKeys
from modelscope.preprocessors.base import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.config import Config, ConfigFields
from modelscope.utils.constant import Fields, InputFields, ModeKeys, ModelFile
from modelscope.utils.hub import get_model_type, parse_label_mapping
from modelscope.utils.logger import get_logger
from modelscope.utils.nlp import import_external_nltk_data
from modelscope.utils.type_assert import type_assert


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.mglm_summarization)
class MGLMSummarizationPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        """preprocess the data
        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)

    @type_assert(object, (str, tuple, Dict))
    def __call__(self, data: Union[str, tuple, Dict]) -> Dict[str, Any]:
        return data
