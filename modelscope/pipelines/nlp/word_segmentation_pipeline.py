from typing import Any, Dict, Optional, Union

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.nlp import StructBertForTokenClassification
from modelscope.preprocessors import TokenClassifcationPreprocessor
from modelscope.utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['WordSegmentationPipeline']


@PIPELINES.register_module(
    Tasks.word_segmentation, module_name=Pipelines.word_segmentation)
class WordSegmentationPipeline(Pipeline):

    def __init__(self,
                 model: Union[StructBertForTokenClassification, str],
                 preprocessor: Optional[TokenClassifcationPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp word segmentation pipeline for prediction

        Args:
            model (StructBertForTokenClassification): a model instance
            preprocessor (TokenClassifcationPreprocessor): a preprocessor instance
        """
        model = model if isinstance(
            model,
            StructBertForTokenClassification) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TokenClassifcationPreprocessor(model.model_dir)
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = preprocessor.tokenizer
        self.config = model.config
        self.id2label = self.config.id2label

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """

        pred_list = inputs['predictions']
        labels = []
        for pre in pred_list:
            labels.append(self.id2label[pre])
        labels = labels[1:-1]
        chunks = []
        chunk = ''
        assert len(inputs['text']) == len(labels)
        for token, label in zip(inputs['text'], labels):
            if label[0] == 'B' or label[0] == 'I':
                chunk += token
            else:
                chunk += token
                chunks.append(chunk)
                chunk = ''
        if chunk:
            chunks.append(chunk)
        seg_result = ' '.join(chunks)
        rst = {
            'output': seg_result,
        }
        return rst
