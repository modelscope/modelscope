from typing import Dict, Optional, Union

from ...metainfo import Pipelines
from ...models import Model
from ...models.nlp import PalmForTextGeneration
from ...preprocessors import TextGenerationPreprocessor
from ...utils.constant import Tasks
from ..base import Pipeline, Tensor
from ..builder import PIPELINES

__all__ = ['TextGenerationPipeline']


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.text_generation)
class TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[PalmForTextGeneration, str],
                 preprocessor: Optional[TextGenerationPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text classification pipeline for prediction

        Args:
            model (SequenceClassificationModel): a model instance
            preprocessor (SequenceClassificationPreprocessor): a preprocessor instance
        """
        model = model if isinstance(
            model, PalmForTextGeneration) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TextGenerationPreprocessor(
                model.model_dir,
                model.tokenizer,
                first_sequence='sentence',
                second_sequence=None)
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = model.tokenizer

    def postprocess(self, inputs: Dict[str, Tensor], **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        replace_tokens_bert = (('[unused0]', ''), ('[PAD]', ''),
                               ('[unused1]', ''), (r' +', ' '), ('[SEP]', ''),
                               ('[unused2]', ''), ('[CLS]', ''), ('[UNK]', ''))
        replace_tokens_roberta = ((r' +', ' '), ('<mask>', '<q>'), ('<pad>',
                                                                    ''),
                                  ('<s>', ''), ('</s>', ''), ('<unk>', ' '))

        pred_list = inputs['predictions']
        pred_ids = pred_list[0][0].cpu().numpy().tolist()
        pred_string = self.tokenizer.decode(pred_ids)
        for _old, _new in replace_tokens_bert:
            pred_string = pred_string.replace(_old, _new)
        pred_string.strip()
        for _old, _new in replace_tokens_roberta:
            pred_string = pred_string.replace(_old, _new)
        pred_string.strip()
        return {'text': pred_string}
