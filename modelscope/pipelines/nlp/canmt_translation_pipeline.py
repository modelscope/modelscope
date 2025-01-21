# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict, Optional, Union

import torch
from sacremoses import MosesDetokenizer

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import CanmtForTranslation
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import CanmtTranslationPreprocessor, Preprocessor
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['CanmtTranslationPipeline']


@PIPELINES.register_module(
    Tasks.competency_aware_translation,
    module_name=Pipelines.canmt_translation)
class CanmtTranslationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 **kwargs):
        """Use `model` and `preprocessor` to create a canmt translation pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the canmt translation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

        Examples:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='competency_aware_translation',
            >>>    model='damo/nlp_canmt_translation_zh2en_large')
            >>> sentence1 = '世界是丰富多彩的。'
            >>> print(pipeline_ins(sentence1))
            >>> # Or use the list input:
            >>> print(pipeline_ins([sentence1])

            To view other examples plese check tests/pipelines/test_canmt_translation.py.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)
        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'

        if self.preprocessor is None:
            self.preprocessor = CanmtTranslationPreprocessor(
                self.model.model_dir,
                kwargs) if preprocessor is None else preprocessor
        self.vocab_tgt = self.preprocessor.vocab_tgt
        self.detokenizer = MosesDetokenizer(lang=self.preprocessor.tgt_lang)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        batch_size = len(inputs[0])
        hypos = []
        scores = []
        for i in range(batch_size):
            hypo_tensor = inputs[0][i][0]['tokens']
            score = inputs[1][i][0].cpu().tolist()
            hypo_sent = self.vocab_tgt.string(
                hypo_tensor,
                '@@ ',
                extra_symbols_to_ignore={self.vocab_tgt.pad()})
            hypo_sent = self.detokenizer.detokenize(hypo_sent.split())
            hypos.append(hypo_sent)
            scores.append(score)

        return {OutputKeys.TRANSLATION: hypos, OutputKeys.SCORE: scores}
