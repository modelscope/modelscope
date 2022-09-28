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

    def _evaluate_postprocess(self, ids_list: List[List[int]]) -> List[str]:
        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''), ('[unused1]',
                                                                  ''),
                               (r' +', ' '), ('[SEP]', ''), ('[unused2]', ''),
                               ('[CLS]', ''), ('[UNK]', ''), (' ', ''))
        replace_tokens_roberta = ((r' +', ' '), ('<mask>', '. '),
                                  ('<pad>', ''), ('<s>', ''), ('</s>', ''),
                                  ('<unk>', ' '), ('<q>', '. '))

        replace_tokens = replace_tokens_roberta \
            if self.model.config.encoder == 'roberta' else replace_tokens_bert
        strings = [self.tokenizer.decode(pred_ids) for pred_ids in ids_list]
        for _old, _new in replace_tokens:
            strings = [s.replace(_old, _new) for s in strings]
        return strings

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
                    or
                    {
                        'preds': List["hello word"...] # the predicted strings
                        'tgts': List["hello world"...] # target strings
                    }
        """
        if self.training:
            return self.model(**input)
        else:
            outputs = self.generator(input['input_ids'],
                                     input['attention_mask'])
            preds = outputs['predictions']
            pred_ids_list = [
                pred_batch[0].cpu().numpy().tolist() for pred_batch in preds
            ]
            tgt_ids_list = input['labels'].cpu().numpy().tolist()
            return {
                'preds': self._evaluate_postprocess(pred_ids_list),
                'tgts': self._evaluate_postprocess(tgt_ids_list)
            }

    def generate(self, input: Dict[str, Tensor]) -> Dict[str, str]:
        outputs = self.generator(**input)
        preds = outputs['predictions']
        pred_ids_list = [preds[0][0].cpu().numpy().tolist()]
        return {OutputKeys.TEXT: self._evaluate_postprocess(pred_ids_list)[0]}
