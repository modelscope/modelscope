import os
from typing import Any, Dict, Optional, Union

import torch

from modelscope.outputs import OutputKeys
from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp.masked_language import MaskedLanguageModelBase
from ...preprocessors import FillMaskPreprocessor
from ...utils.config import Config
from ...utils.constant import ModelFile, Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['FillMaskPipeline']
_type_map = {'veco': 'roberta', 'sbert': 'bert'}


@PIPELINES.register_module(Tasks.fill_mask, module_name=Pipelines.fill_mask)
class FillMaskPipeline(Pipeline):

    def __init__(self,
                 model: Union[MaskedLanguageModelBase, str],
                 preprocessor: Optional[FillMaskPreprocessor] = None,
                 first_sequence='sentense',
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp fill mask pipeline for prediction

        Args:
            model (MaskedLanguageModelBase): a model instance
            preprocessor (FillMaskPreprocessor): a preprocessor instance
        """
        fill_mask_model = model if isinstance(
            model, MaskedLanguageModelBase) else Model.from_pretrained(model)

        if preprocessor is None:
            preprocessor = FillMaskPreprocessor(
                fill_mask_model.model_dir,
                first_sequence=first_sequence,
                second_sequence=None)
        fill_mask_model.eval()
        super().__init__(
            model=fill_mask_model, preprocessor=preprocessor, **kwargs)

        self.preprocessor = preprocessor
        self.config = Config.from_file(
            os.path.join(fill_mask_model.model_dir, ModelFile.CONFIGURATION))
        self.tokenizer = preprocessor.tokenizer
        self.mask_id = {'roberta': 250001, 'bert': 103}

        self.rep_map = {
            'bert': {
                '[unused0]': '',
                '[PAD]': '',
                '[unused1]': '',
                r' +': ' ',
                '[SEP]': '',
                '[unused2]': '',
                '[CLS]': '',
                '[UNK]': ''
            },
            'roberta': {
                r' +': ' ',
                '<mask>': '<q>',
                '<pad>': '',
                '<s>': '',
                '</s>': '',
                '<unk>': ' '
            }
        }

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        import numpy as np
        logits = inputs['logits'].detach().numpy()
        input_ids = inputs['input_ids'].detach().numpy()
        pred_ids = np.argmax(logits, axis=-1)
        model_type = self.model.config.model_type
        process_type = model_type if model_type in self.mask_id else _type_map[
            model_type]
        rst_ids = np.where(input_ids == self.mask_id[process_type], pred_ids,
                           input_ids)

        def rep_tokens(string, rep_map):
            for k, v in rep_map.items():
                string = string.replace(k, v)
            return string.strip()

        pred_strings = []
        for ids in rst_ids:  # batch
            if 'language' in self.config.model and self.config.model.language == 'zh':
                pred_string = self.tokenizer.convert_ids_to_tokens(ids)
                pred_string = ''.join(pred_string)
            else:
                pred_string = self.tokenizer.decode(ids)
            pred_string = rep_tokens(pred_string, self.rep_map[process_type])
            pred_strings.append(pred_string)

        return {OutputKeys.TEXT: pred_strings}
