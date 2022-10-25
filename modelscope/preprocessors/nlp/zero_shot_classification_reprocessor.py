# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict, Union

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from .nlp_base import NLPTokenizerPreprocessorBase


@PREPROCESSORS.register_module(
    Fields.nlp, module_name=Preprocessors.zero_shot_cls_tokenizer)
class ZeroShotClassificationPreprocessor(NLPTokenizerPreprocessorBase):
    """The tokenizer preprocessor used in zero shot classification.
    """

    def __init__(self, model_dir: str, mode=ModeKeys.INFERENCE, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        self.sequence_length = kwargs.pop('sequence_length', 512)
        super().__init__(model_dir, mode=mode, **kwargs)

    def __call__(self, data: Union[str, Dict], hypothesis_template: str,
                 candidate_labels: list) -> Dict[str, Any]:
        """process the raw input data

        Args:
            data (str or dict): a sentence
                Example:
                    'you are so handsome.'

        Returns:
            Dict[str, Any]: the preprocessed data
        """
        if isinstance(data, dict):
            data = data.get(self.first_sequence)

        pairs = [[data, hypothesis_template.format(label)]
                 for label in candidate_labels]

        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.sequence_length,
            truncation_strategy='only_first',
            return_tensors='pt' if self._mode == ModeKeys.INFERENCE else None)
        return features
