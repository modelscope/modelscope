from typing import Dict, Optional, Union

from modelscope.models import Model
from modelscope.models.nlp import PalmForTextGenerationModel
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['TextGenerationPipeline']


@PIPELINES.register_module(Tasks.text_generation, module_name=r'palm')
class TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[PalmForTextGenerationModel, str],
                 preprocessor: Optional[TextGenerationPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """
        sc_model = model if isinstance(
            model,
            PalmForTextGenerationModel) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TextGenerationPreprocessor(
                sc_model.model_dir,
                first_sequence='sentence',
                second_sequence=None)
        super().__init__(model=sc_model, preprocessor=preprocessor, **kwargs)
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
