from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import TextGenerationPreprocessor
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

__all__ = ['TextGenerationPipeline']


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.text_generation)
class TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[TextGenerationPreprocessor] = None,
                 first_sequence='sentence',
                 **kwargs):
        """Use `model` and `preprocessor` to create a generation pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the text generation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            first_sequence: The key to read the first sentence in.
            sequence_length: Max sequence length in the user's custom scenario. 128 will be used as a default value.

            NOTE: Inputs of type 'str' are also supported. In this scenario, the 'first_sequence'
            param will have no effect.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='text-generation',
            >>>    model='damo/nlp_palm2.0_text-generation_chinese-base')
            >>> sentence1 = '本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：'
            >>>     '1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代'
            >>> print(pipeline_ins(sentence1))
            >>> # Or use the dict input:
            >>> print(pipeline_ins({'sentence': sentence1}))

            To view other examples plese check the tests/pipelines/test_text_generation.py.
        """
        model = model if isinstance(model,
                                    Model) else Model.from_pretrained(model)
        if preprocessor is None:
            preprocessor = TextGenerationPreprocessor(
                model.model_dir,
                first_sequence=first_sequence,
                second_sequence=None,
                sequence_length=kwargs.pop('sequence_length', 128))
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.tokenizer = preprocessor.tokenizer

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
        generate_context = inputs["generate_context"]
        generate_context = "".join(self.tokenizer.convert_ids_to_tokens(generate_context)).replace('[UNK]', '“').replace('##','')
        return {OutputKeys.TEXT: generate_context}
