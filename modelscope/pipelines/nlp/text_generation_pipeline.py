from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['TextGenerationPipeline']


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.text_generation)
class TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[TextGenerationPreprocessor] = None,
                 **kwargs):
        """use `model` and `preprocessor` to create a nlp text generation pipeline for prediction

        Args:
            model (PalmForTextGeneration): a model instance
            preprocessor (TextGenerationPreprocessor): a preprocessor instance
        """
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TextGenerationPreprocessor(
                model.model_dir,
                first_sequence='sentence',
                second_sequence=None,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model.generate(inputs)

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        return inputs
