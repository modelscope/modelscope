from typing import Dict

from ...metainfo import Models
from ...utils.constant import Tasks
from ..base import Tensor, TorchModel
from ..builder import MODELS

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

        from sofa.models.palm_v2 import PalmForConditionalGeneration, Translator
        self.model = PalmForConditionalGeneration.from_pretrained(model_dir)
        self.tokenizer = self.model.tokenizer
        self.generator = Translator(self.model)

    def _evaluate_postprocess(self, src: Tensor, tgt: Tensor,
                              mask_src: Tensor) -> Dict[str, str]:
        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''),
                               ('[unused1]', ''), (r' +', ' '), ('[SEP]', ''),
                               ('[unused2]', ''), ('[CLS]', ''), ('[UNK]', ''))
        replace_tokens_roberta = ((r' +', ' '), ('<mask>', '<q>'), ('<pad>',
                                                                    ''),
                                  ('<s>', ''), ('</s>', ''), ('<unk>', ' '))

        inputs = self.generator(src, mask_src)
        pred_list = inputs['predictions']
        pred_id_list = [
            pred_batch[0].cpu().numpy().tolist() for pred_batch in pred_list
        ]
        tgt_id_list = tgt.cpu().numpy().tolist()
        pred_strings = [
            self.tokenizer.decode(pred_ids) for pred_ids in pred_id_list
        ]
        tgt_strings = [
            self.tokenizer.decode(tgt_ids) for tgt_ids in tgt_id_list
        ]
        for _old, _new in replace_tokens_bert:
            pred_strings = [s.replace(_old, _new) for s in pred_strings]
            tgt_strings = [s.replace(_old, _new) for s in tgt_strings]
        for _old, _new in replace_tokens_roberta:
            pred_strings = [s.replace(_old, _new) for s in pred_strings]
            tgt_strings = [s.replace(_old, _new) for s in tgt_strings]
        for s in pred_strings:
            s.strip()
        for s in tgt_strings:
            s.strip()
        return {'preds': pred_strings, 'tgts': tgt_strings}

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'predictions': Tensor([[1377, 4959, 2785, 6392...])]), # tokens need to be decode by tokenizer
                    }
        """
        if self.training:
            return {'loss': self.model(**input)}
        elif 'tgt' in input:
            return self._evaluate_postprocess(**input)
        else:
            return self.generator(**input)
