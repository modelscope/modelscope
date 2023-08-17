# Copyright (c) Alibaba, Inc. and its affiliates.

# Copyright (c) 2022 Zhipu.AI
import os
from typing import Any, Dict, Optional, Union

import torch
from transformers import GenerationConfig

from modelscope import snapshot_download
from modelscope.metainfo import Pipelines
from modelscope.models.base import Model
from modelscope.outputs import (ModelOutputBase, OutputKeys,
                                TokenGeneratorOutput)
from modelscope.pipelines.base import Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import Preprocessor
from modelscope.utils.chinese_utils import remove_space_between_chinese_chars
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.hub import Config, read_config
from modelscope.utils.streaming_output import PipelineStreamingOutputMixin

__all__ = [
    'TextGenerationPipeline',
    'TextGenerationT5Pipeline',
    'ChatGLM6bTextGenerationPipeline',
    'ChatGLM6bV2TextGenerationPipeline',
    'QWenChatPipeline',
    'QWenTextGenerationPipeline',
]


@PIPELINES.register_module(
    Tasks.text_generation, module_name=Pipelines.text_generation)
class TextGenerationPipeline(Pipeline, PipelineStreamingOutputMixin):

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

        Examples:
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
            auto_collate=auto_collate,
            compile=kwargs.pop('compile', False),
            compile_options=kwargs.pop('compile_options', {}),
            **kwargs)

        assert isinstance(self.model, Model), \
            f'please check whether model config exists in {ModelFile.CONFIGURATION}'
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


@PIPELINES.register_module(
    group_key=Tasks.chat, module_name='chatglm6b-text-generation')
class ChatGLM6bTextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 quantization_bit=None,
                 use_bf16=False,
                 **kwargs):
        from modelscope.models.nlp.chatglm.text_generation import (
            ChatGLMConfig, ChatGLMForConditionalGeneration)
        if isinstance(model, str):
            model_dir = snapshot_download(
                model) if not os.path.exists(model) else model
            model = ChatGLMForConditionalGeneration.from_pretrained(
                model_dir).half()
            if torch.cuda.is_available():
                model = model.cuda()
        if quantization_bit is not None:
            model = model.quantize(quantization_bit)
        if use_bf16:
            model = model.bfloat16()
        self.model = model
        self.model.eval()

        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        inputs.update(forward_params)
        return self.model.chat(inputs)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    group_key=Tasks.chat, module_name='chatglm2_6b-text-generation')
class ChatGLM6bV2TextGenerationPipeline(Pipeline):

    def __init__(self,
                 model: Union[Model, str],
                 quantization_bit=None,
                 use_bf16=False,
                 **kwargs):
        from modelscope import AutoModel, AutoTokenizer
        device: str = kwargs.get('device', 'gpu')
        if isinstance(model, str):
            revision = kwargs.get('revision', None)
            model_dir = snapshot_download(
                model,
                revision=revision) if not os.path.exists(model) else model
            default_device_map = None
            if device.startswith('gpu') or device.startswith('cuda'):
                default_device_map = {'': 0}
            device_map = kwargs.get('device_map', default_device_map)
            default_torch_dtype = None
            if use_bf16:
                default_torch_dtype = torch.bfloat16
            torch_dtype = kwargs.get('torch_dtype', default_torch_dtype)
            model = AutoModel.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch_dtype)
        else:
            if device.startswith('gpu') or device.startswith('cuda'):
                model.cuda()
            if use_bf16:
                model.bfloat16()
        if quantization_bit is not None:
            model = model.quantize(quantization_bit)

        self.model = model
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model.model_dir, trust_remote_code=True)

        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: Dict, **forward_params) -> Dict[str, Any]:
        query = inputs['text']
        history = inputs['history']
        if isinstance(history, torch.Tensor):
            history = history.tolist()
        result = self.model.chat(self.tokenizer, query, history,
                                 **forward_params)
        return {'response': result[0], 'history': result[1]}

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(group_key=Tasks.chat, module_name='qwen-chat')
class QWenChatPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], **kwargs):
        from modelscope.models.nlp import (QWenConfig, QWenForTextGeneration,
                                           QWenTokenizer)
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        use_max_memory = kwargs.get('use_max_memory', False)
        quantization_config = kwargs.get('quantization_config', None)

        if use_max_memory:
            max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
        else:
            max_memory = None

        if isinstance(model, str):
            model_dir = snapshot_download(
                model) if not os.path.exists(model) else model

            config = read_config(model_dir)
            model_config = QWenConfig.from_pretrained(model_dir)
            model_config.torch_dtype = torch_dtype

            model = QWenForTextGeneration.from_pretrained(
                model_dir,
                cfg_dict=config,
                config=model_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                max_memory=max_memory)
            model.generation_config = GenerationConfig.from_pretrained(
                model_dir)

        self.model = model
        self.model.eval()
        self.tokenizer = QWenTokenizer.from_pretrained(self.model.model_dir)

        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: str, **forward_params) -> Dict[str, Any]:
        history = forward_params.get('history', None)
        system = forward_params.get('system', 'You are a helpful assistant.')
        append_history = forward_params.get('append_history', True)
        return self.model.chat(self.tokenizer, inputs, history, system,
                               append_history)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    group_key=Tasks.text_generation, module_name='qwen-text-generation')
class QWenTextGenerationPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], **kwargs):
        from modelscope.models.nlp import (QWenConfig, QWenForTextGeneration,
                                           QWenTokenizer)
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        use_max_memory = kwargs.get('use_max_memory', False)
        quantization_config = kwargs.get('quantization_config', None)

        if use_max_memory:
            max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
        else:
            max_memory = None

        if isinstance(model, str):
            model_dir = snapshot_download(
                model) if not os.path.exists(model) else model

            config = read_config(model_dir)
            model_config = QWenConfig.from_pretrained(model_dir)
            model_config.torch_dtype = torch_dtype

            model = QWenForTextGeneration.from_pretrained(
                model_dir,
                cfg_dict=config,
                config=model_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                max_memory=max_memory)
            model.generation_config = GenerationConfig.from_pretrained(
                model_dir)

        self.model = model
        self.model.eval()
        self.tokenizer = QWenTokenizer.from_pretrained(self.model.model_dir)

        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: str, **forward_params) -> Dict[str, Any]:
        return {
            OutputKeys.TEXT:
            self.model.chat(self.tokenizer, inputs,
                            history=None)[OutputKeys.RESPONSE]
        }

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
