from typing import Dict

from modelscope.models.nlp import MaskedLanguageModel
from modelscope.preprocessors import FillMaskPreprocessor
from modelscope.utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['FillMaskPipeline']


@PIPELINES.register_module(Tasks.fill_mask, module_name=r'sbert')
@PIPELINES.register_module(Tasks.fill_mask, module_name=r'veco')
class FillMaskPipeline(Pipeline):

    def __init__(self, model: MaskedLanguageModel,
                 preprocessor: FillMaskPreprocessor, **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.preprocessor = preprocessor
        self.tokenizer = preprocessor.tokenizer
        self.mask_id = {'veco': 250001, 'sbert': 103}

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
        rst_ids = np.where(
            input_ids == self.mask_id[self.model.config.model_type], pred_ids,
            input_ids)
        pred_strings = []
        for ids in rst_ids:
            if self.model.config.model_type == 'veco':
                pred_string = self.tokenizer.decode(ids).split(
                    '</s>')[0].replace('<s>',
                                       '').replace('</s>',
                                                   '').replace('<pad>', '')
            elif self.model.config.vocab_size == 21128:  # zh bert
                pred_string = self.tokenizer.convert_ids_to_tokens(ids)
                pred_string = ''.join(pred_string).replace('##', '')
                pred_string = pred_string.split('[SEP]')[0].replace(
                    '[CLS]', '').replace('[SEP]', '').replace('[UNK]', '')
            else:  ## en bert
                pred_string = self.tokenizer.decode(ids)
                pred_string = pred_string.split('[SEP]')[0].replace(
                    '[CLS]', '').replace('[SEP]', '').replace('[UNK]', '')
            pred_strings.append(pred_string)

        return {'pred_string': pred_strings}
