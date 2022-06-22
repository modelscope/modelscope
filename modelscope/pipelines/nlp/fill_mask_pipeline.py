from typing import Dict, Optional, Union, Any

import torch

from ...models import Model
from ...models.nlp.masked_language_model import \
    MaskedLMModelBase
from ...preprocessors import FillMaskPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES
from ...metainfo import Pipelines

__all__ = ['FillMaskPipeline']


@PIPELINES.register_module(Tasks.fill_mask, module_name=Pipelines.fill_mask)
class FillMaskPipeline(Pipeline):

    def __init__(self,
                 model: Union[MaskedLMModelBase, str],
                 preprocessor: Optional[FillMaskPreprocessor] = None,
                 first_sequence="sentense",
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp fill mask pipeline for prediction

        Args:
            model (MaskedLMModelBase): a model instance
            preprocessor (FillMaskPreprocessor): a preprocessor instance
        """
        fill_mask_model = model if isinstance(
            model, MaskedLMModelBase) else Model.from_pretrained(model)
        assert fill_mask_model.config is not None

        if preprocessor is None:
            preprocessor = FillMaskPreprocessor(
                fill_mask_model.model_dir,
                first_sequence=first_sequence,
                second_sequence=None)
        fill_mask_model.eval()
        super().__init__(model=fill_mask_model, preprocessor=preprocessor, **kwargs)
        self.preprocessor = preprocessor
        self.tokenizer = preprocessor.tokenizer
        self.mask_id = {'veco': 250001, 'sbert': 103}

        self.rep_map = {
            'sbert': {
                '[unused0]': '',
                '[PAD]': '',
                '[unused1]': '',
                r' +': ' ',
                '[SEP]': '',
                '[unused2]': '',
                '[CLS]': '',
                '[UNK]': ''
            },
            'veco': {
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
        rst_ids = np.where(input_ids == self.mask_id[model_type], pred_ids,
                           input_ids)

        def rep_tokens(string, rep_map):
            for k, v in rep_map.items():
                string = string.replace(k, v)
            return string.strip()

        pred_strings = []
        for ids in rst_ids:  # batch
            # TODO vocab size is not stable
            if self.model.config.vocab_size == 21128:  # zh bert
                pred_string = self.tokenizer.convert_ids_to_tokens(ids)
                pred_string = ''.join(pred_string)
            else:
                pred_string = self.tokenizer.decode(ids)
            pred_string = rep_tokens(pred_string, self.rep_map[model_type])
            pred_strings.append(pred_string)

        return {'text': pred_strings}
