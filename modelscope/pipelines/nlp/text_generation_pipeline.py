# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Any, Dict, Optional, Union

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import (ModelOutputBase, OutputKeys,
                                TokenGeneratorOutput)
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.chinese_utils import remove_space_between_chinese_chars
from modelscope.utils.constant import Tasks
from modelscope.utils.hub import Config, read_config

__all__ = ['TextGenerationPipeline', 'TextGenerationT5Pipeline']


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.text_generation)
class TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
                 first_sequence='sentence',
                 **kwargs):
        """Use `model` and `preprocessor` to create a generation pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the text generation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.

            Example:
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline(task='text-generation',
            >>>    model='damo/nlp_palm2.0_text-generation_chinese-base')
            >>> sentence1 = '本文总结了十个可穿戴产品的设计原则，而这些原则，同样也是笔者认为是这个行业最吸引人的地方：'
            >>>     '1.为人们解决重复性问题；2.从人开始，而不是从机器开始；3.要引起注意，但不要刻意；4.提升用户能力，而不是取代'
            >>> print(pipeline_ins(sentence1))
            >>> # Or use the dict input:
            >>> print(pipeline_ins({'sentence': sentence1}))

            To view other examples plese check tests/pipelines/test_text_generation.py.
        """
        super().__init__(
            model=model,
            preprocessor=preprocessor,
            config_file=config_file,
            device=device,
            auto_collate=auto_collate)

        if preprocessor is None:
            self.preprocessor = Preprocessor.from_pretrained(
                self.model.model_dir, first_sequence=first_sequence, **kwargs)
        self.model.eval()
        self.postprocessor = kwargs.pop('postprocessor', None)
        if self.postprocessor is None and hasattr(self.model, 'model_dir'):
            # Compatible with old code
            cfg = read_config(self.model.model_dir)
            self.postprocessor = cfg.get('postprocessor')
        if self.postprocessor is None:
            self.postprocessor = 'decode'

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            return self.model.generate(inputs, **forward_params)

    def decode(self, inputs) -> str:
        return self.preprocessor.decode(
            inputs.tolist(), skip_special_tokens=True)

    def sentence_piece(self, inputs) -> str:
        return self.preprocessor.decode(inputs.tolist())

    def roberta(self, inputs) -> str:
        decoded = self.preprocessor.decode(inputs.tolist())
        return decoded.replace('<q>', '. ').replace('<mask>',
                                                    '. ').replace('</s>', '')

    def postprocess(self, inputs: Union[Dict[str, Tensor],
                                        TokenGeneratorOutput],
                    **postprocess_params) -> Dict[str, str]:
        """process the prediction results

        Args:
            inputs (Dict[str, Any]): _description_

        Returns:
            Dict[str, str]: the prediction results
        """
        if isinstance(inputs, (dict, ModelOutputBase)):
            inputs = inputs['sequences']
        if isinstance(inputs, list) or len(inputs.shape) > 1:
            inputs = inputs[0]
        decoded = getattr(self, self.postprocessor)(inputs)
        text = remove_space_between_chinese_chars(decoded)
        return {OutputKeys.TEXT: text}


@PIPELINES.register_module(
    Tasks.text2text_generation, module_name=Pipelines.translation_en_to_de)
@PIPELINES.register_module(
    Tasks.text2text_generation, module_name=Pipelines.translation_en_to_ro)
@PIPELINES.register_module(
    Tasks.text2text_generation, module_name=Pipelines.translation_en_to_fr)
@PIPELINES.register_module(
    Tasks.text2text_generation, module_name=Pipelines.text2text_generation)
class TextGenerationT5Pipeline(TextGenerationPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Optional[Preprocessor] = None,
                 sub_task=None,
                 **kwargs):
        super().__init__(model, preprocessor, **kwargs)
        self.sub_task = sub_task
        self.task_specific_params = self._parse_specific_model_params(
            getattr(self.model, 'model_dir', None), 'task_specific_params')
        self.min_length = self._parse_specific_model_params(
            getattr(self.model, 'model_dir', None), 'min_length')
        self.max_length = self._parse_specific_model_params(
            getattr(self.model, 'model_dir', None), 'max_length')

    def _parse_specific_model_params(self, model_dir, key):
        if model_dir is None:
            return

        cfg: Config = read_config(model_dir)
        params = cfg.safe_get(f'model.{key}')
        if params is None:
            cfg: Config = read_config(os.path.join(model_dir, 'config.json'))
            params = cfg.safe_get(key)
        return params

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        if not isinstance(inputs, str):
            raise ValueError(f'Not supported input type: {type(inputs)}')

        if self.task_specific_params is not None:
            sub_task = self.sub_task or self.model.pipeline.type
            if sub_task in self.task_specific_params:
                self.model.config.update(self.task_specific_params[sub_task])
                if 'prefix' in self.task_specific_params[sub_task]:
                    inputs = self.task_specific_params[sub_task].prefix + inputs

        return super().preprocess(inputs, **preprocess_params)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:

        min_length = forward_params.get('min_length', self.min_length)
        max_length = forward_params.get('max_length', self.max_length)
        if min_length is not None:
            forward_params['min_length'] = min_length
        if max_length is not None:
            forward_params['max_length'] = max_length

        with torch.no_grad():
            return self.model.generate(**inputs, **forward_params)
