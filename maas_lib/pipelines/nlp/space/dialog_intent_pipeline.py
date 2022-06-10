from typing import Any, Dict, Optional

from maas_lib.models.nlp import DialogIntentModel
from maas_lib.preprocessors import DialogIntentPreprocessor
from maas_lib.utils.constant import Tasks
from ...base import Model, Tensor
from ...builder import PIPELINES

__all__ = ['DialogIntentPipeline']


@PIPELINES.register_module(Tasks.dialog_intent, module_name=r'space-intent')
class DialogIntentPipeline(Model):

    def __init__(self, model: DialogIntentModel,
                 preprocessor: DialogIntentPreprocessor, **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """

        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model = model
        self.tokenizer = preprocessor.tokenizer

    def postprocess(self, inputs: Dict[str, Tensor]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        vocab_size = len(self.tokenizer.vocab)
        pred_list = inputs['predictions']
        pred_ids = pred_list[0][0].cpu().numpy().tolist()
        for j in range(len(pred_ids)):
            if pred_ids[j] >= vocab_size:
                pred_ids[j] = 100
        pred = self.tokenizer.convert_ids_to_tokens(pred_ids)
        pred_string = ''.join(pred).replace(
            '##',
            '').split('[SEP]')[0].replace('[CLS]',
                                          '').replace('[SEP]',
                                                      '').replace('[UNK]', '')
        return {'pred_string': pred_string}
