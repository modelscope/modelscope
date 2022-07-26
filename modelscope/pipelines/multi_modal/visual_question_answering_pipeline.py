from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.multi_modal import MPlugForVisualQuestionAnswering
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import MPlugVisualQuestionAnsweringPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['VisualQuestionAnsweringPipeline']


@PIPELINES.register_module(
    Tasks.visual_question_answering,
    module_name=Pipelines.visual_question_answering)
class VisualQuestionAnsweringPipeline(Pipeline):

    def __init__(self,
                 model: Union[MPlugForVisualQuestionAnswering, str],
                 preprocessor: Optional[
                     MPlugVisualQuestionAnsweringPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a visual question answering pipeline for prediction

        Args:
            model (MPlugForVisualQuestionAnswering): a model instance
            preprocessor (MPlugVisualQuestionAnsweringPreprocessor): a preprocessor instance
        """
        model = model if isinstance(
            model,
            MPlugForVisualQuestionAnswering) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = MPlugVisualQuestionAnsweringPreprocessor(
                model.model_dir)
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = model.tokenizer

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return super().forward(inputs, **forward_params)

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''),
                               ('[unused1]', ''), (r' +', ' '), ('[SEP]', ''),
                               ('[unused2]', ''), ('[CLS]', ''), ('[UNK]', ''))

        pred_string = self.tokenizer.decode(inputs[0][0])
        for _old, _new in replace_tokens_bert:
            pred_string = pred_string.replace(_old, _new)
        pred_string.strip()
        return {OutputKeys.TEXT: pred_string}
