# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Text2TextGenerationPreprocessor
from modelscope.utils.constant import Tasks

__all__ = ['Text2TextGenerationPipeline']


@PIPELINES.register_module(
    Tasks.text2text_generation, module_name=Pipelines.text2text_generation)
class Text2TextGenerationPipeline(Pipeline):

    def __init__(
            self,
            model: Union[Model, str],
            preprocessor: Optional[Text2TextGenerationPreprocessor] = None,
            first_sequence='sentence',
            **kwargs):
        """Use `model` and `preprocessor` to create a text to text generation pipeline for prediction.

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
            preprocessor = Text2TextGenerationPreprocessor(
                model.model_dir,
                sequence_length=kwargs.pop('sequence_length', 128))
        self.tokenizer = preprocessor.tokenizer
        model.eval()
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:

        forward_params['min_length'] = forward_params.get(
            'min_length', self.model.config.min_length)
        forward_params['max_length'] = forward_params.get(
            'max_length', self.model.config.max_length)

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **forward_params)
            return {'output_ids': output_ids}

    def postprocess(self, inputs: Dict[str, Tensor],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        output = self.tokenizer.decode(
            inputs['output_ids'][0],
            skip_special_tokens=True,
        )
        return {OutputKeys.TEXT: output}
