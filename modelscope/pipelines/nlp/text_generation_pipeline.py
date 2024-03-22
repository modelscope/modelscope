# Copyright (c) Alibaba, Inc. and its affiliates.

# Copyright (c) 2022 Zhipu.AI
import os
from typing import Any, Dict, List, Optional, Union

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
from modelscope.utils.logger import get_logger
from modelscope.utils.streaming_output import PipelineStreamingOutputMixin
from modelscope.utils.torch_utils import is_on_same_device

logger = get_logger()

__all__ = [
    'TextGenerationPipeline', 'TextGenerationT5Pipeline',
    'ChatGLM6bTextGenerationPipeline', 'ChatGLM6bV2TextGenerationPipeline',
    'QWenChatPipeline', 'QWenTextGenerationPipeline', 'SeqGPTPipeline',
    'Llama2TaskPipeline'
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
        self.has_logged = False

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self, inputs: Union[Dict[str, Any], Tensor],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            try:
                return self.model.generate(inputs, **forward_params)
            except AttributeError as e:
                if not self.has_logged:
                    logger.warning(
                        'When inputs are passed directly, '
                        f'the error is {e}, '
                        'which can be ignored if it runs correctly.')
                    self.has_logged = True
                return self.model.generate(**inputs, **forward_params)

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
            ChatGLMForConditionalGeneration)
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
        from modelscope import AutoTokenizer
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
            model = Model.from_pretrained(
                model_dir,
                trust_remote_code=True,
                device_map=device_map,
                torch_dtype=torch_dtype)
        else:
            if ((device.startswith('gpu') or device.startswith('cuda'))
                    and is_on_same_device(model)):
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
        inputs.update(forward_params)
        return self.model.chat(inputs, self.tokenizer)

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(group_key=Tasks.chat, module_name='qwen-chat')
class QWenChatPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], **kwargs):
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        use_max_memory = kwargs.get('use_max_memory', False)
        revision = kwargs.get('model_revision', 'v.1.0.5')

        if use_max_memory:
            max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
        else:
            max_memory = None
        if torch_dtype == 'bf16' or torch_dtype == torch.bfloat16:
            bf16 = True
        else:
            bf16 = False

        if isinstance(model, str):
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, revision=revision, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map=device_map,
                revision=revision,
                trust_remote_code=True,
                fp16=bf16).eval()
            self.model.generation_config = GenerationConfig.from_pretrained(
                model, trust_remote_code=True)  # 可指定不同的生成长度、top_p等相关超参

        super().__init__(model=self.model, **kwargs)
        # skip pipeline model placement
        self._model_prepare = True

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: Union[Dict, str],
                **forward_params) -> Dict[str, Any]:
        if isinstance(inputs, Dict):
            text = inputs.get('text', None)
            history = inputs.get('history', None)
        else:
            text = inputs
            history = forward_params.get('history', None)
        system = forward_params.get('system', 'You are a helpful assistant.')
        append_history = forward_params.get('append_history', True)
        res = self.model.chat(self.tokenizer, text, history, system,
                              append_history)
        return {'response': res[0], 'history': res[1]}

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    group_key=Tasks.text_generation, module_name='qwen-text-generation')
class QWenTextGenerationPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], **kwargs):
        from modelscope import AutoModelForCausalLM, AutoTokenizer
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        use_max_memory = kwargs.get('use_max_memory', False)
        revision = kwargs.get('model_revision', 'v.1.0.4')

        if use_max_memory:
            max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
            n_gpus = torch.cuda.device_count()
            max_memory = {i: max_memory for i in range(n_gpus)}
        else:
            max_memory = None
        if torch_dtype == 'bf16' or torch_dtype == torch.bfloat16:
            bf16 = True
        else:
            bf16 = False

        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map=device_map,
                revision=revision,
                trust_remote_code=True,
                bf16=bf16).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, revision=revision, trust_remote_code=True)
            self.model.generation_config = GenerationConfig.from_pretrained(
                model)
        else:
            self.model = model
            self.tokenizer = kwargs.get('tokenizer', None)

        super().__init__(model=self.model, **kwargs)
        # skip pipeline model placement
        self._model_prepare = True

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, inputs: str, **forward_params) -> Dict[str, Any]:
        inputs = self.tokenizer(inputs, return_tensors='pt').to('cuda:0')
        return {
            OutputKeys.TEXT:
            self.tokenizer.decode(
                self.model.generate(**inputs).cpu()[0],
                skip_special_tokens=True)
        }

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    group_key=Tasks.text_generation, module_name='seqgpt')
class SeqGPTPipeline(Pipeline):

    def __init__(self, model: Union[Model, str], **kwargs):
        from modelscope.utils.hf_util import AutoTokenizer

        if isinstance(model, str):
            model_dir = snapshot_download(
                model) if not os.path.exists(model) else model
            model = Model.from_pretrained(model_dir)
        self.model = model
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        super().__init__(model=model, **kwargs)

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    # define the forward pass
    def forward(self, prompt: str, **forward_params) -> Dict[str, Any]:
        # gen & decode
        # prompt += '[GEN]'
        input_ids = self.tokenizer(
            prompt + forward_params.get('gen_token', ''),
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=1024)
        input_ids = input_ids.input_ids.to(self.model.device)
        outputs = self.model.generate(
            input_ids, num_beams=4, do_sample=False, max_new_tokens=256)
        decoded_sentences = self.tokenizer.batch_decode(
            outputs, skip_special_tokens=True)
        decoded_sentence = decoded_sentences[0]
        decoded_sentence = decoded_sentence[len(prompt):]
        return {OutputKeys.TEXT: decoded_sentence}

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    Tasks.text_generation,
    module_name=Pipelines.llama2_text_generation_pipeline)
class Llama2TaskPipeline(TextGenerationPipeline):

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate=True,
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
            >>> from modelscope.utils.constant import Tasks
            >>> import torch
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope import snapshot_download, Model
            >>> model_dir = snapshot_download("modelscope/Llama-2-13b-chat-ms",
            >>>     ignore_file_pattern = [r'\\w+\\.safetensors'])
            >>> pipe = pipeline(task=Tasks.text_generation, model=model_dir, device_map='auto',
            >>>     torch_dtype=torch.float16)
            >>> inputs="咖啡的作用是什么？"
            >>> result = pipe(inputs,max_length=200, do_sample=True, top_p=0.85,
            >>>     temperature=1.0, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
            >>> print(result['text'])

            To view other examples plese check tests/pipelines/test_llama2_text_generation_pipeline.py.
        """
        self.model = Model.from_pretrained(
            model, device_map='auto', torch_dtype=torch.float16)
        from modelscope.models.nlp.llama2 import Llama2Tokenizer
        self.tokenizer = Llama2Tokenizer.from_pretrained(model)
        super().__init__(model=self.model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self,
                inputs: str,
                max_length: int = 2048,
                do_sample: bool = False,
                top_p: float = 0.9,
                temperature: float = 0.6,
                repetition_penalty: float = 1.,
                eos_token_id: int = 2,
                bos_token_id: int = 1,
                pad_token_id: int = 0,
                **forward_params) -> Dict[str, Any]:
        output = {}
        inputs = self.tokenizer(
            inputs, add_special_tokens=False, return_tensors='pt')
        generate_ids = self.model.generate(
            inputs.input_ids.to('cuda'),
            max_length=max_length,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            eos_token_id=eos_token_id,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
            **forward_params)
        out = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]
        output['text'] = out
        return output

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input


@PIPELINES.register_module(
    Tasks.chat, module_name=Pipelines.llama2_text_generation_chat_pipeline)
class Llama2chatTaskPipeline(Pipeline):
    """Use `model` and `preprocessor` to create a generation pipeline for prediction.

        Args:
            model (str or Model): Supply either a local model dir which supported the text generation task,
            or a model id from the model hub, or a torch model instance.
            preprocessor (Preprocessor): An optional preprocessor instance, please make sure the preprocessor fits for
            the model if supplied.
            kwargs (dict, `optional`):
                Extra kwargs passed into the preprocessor's constructor.
        Examples:
            >>> from modelscope.utils.constant import Tasks
            >>> import torch
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope import Model
            >>> pipe = pipeline(task=Tasks.chat, model="modelscope/Llama-2-7b-chat-ms", device_map='auto',
            >>> torch_dtype=torch.float16, ignore_file_pattern = [r'.+\\.bin$'], model_revision='v1.0.5')
            >>> inputs = 'Where is the capital of Zhejiang?'
            >>> result = pipe(inputs,max_length=512, do_sample=False, top_p=0.9,
            >>> temperature=0.6, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
            >>> print(result['response'])
            >>> inputs = 'What are the interesting places there?'
            >>> result = pipe(inputs,max_length=512, do_sample=False, top_p=0.9,
            >>> temperature=0.6, repetition_penalty=1., eos_token_id=2, bos_token_id=1,
            >>> pad_token_id=0, history=result['history'])
            >>> print(result['response'])
            >>> inputs = 'What are the company there?'
            >>> history_demo = [('Where is the capital of Zhejiang?',
            >>> 'Thank you for asking! The capital of Zhejiang Province is Hangzhou.')]
            >>> result = pipe(inputs,max_length=512, do_sample=False, top_p=0.9,
            >>> temperature=0.6, repetition_penalty=1., eos_token_id=2, bos_token_id=1,
            >>> pad_token_id=0, history=history_demo)
            >>> print(result['response'])

            To view other examples plese check tests/pipelines/test_llama2_text_generation_pipeline.py.
        """

    def __init__(self,
                 model: Union[Model, str],
                 preprocessor: Preprocessor = None,
                 config_file: str = None,
                 device: str = 'gpu',
                 auto_collate: bool = True,
                 **kwargs) -> Dict[str, Any]:
        device_map = kwargs.get('device_map', None)
        torch_dtype = kwargs.get('torch_dtype', None)
        self.model = Model.from_pretrained(
            model, device_map=device_map, torch_dtype=torch_dtype)
        from modelscope.models.nlp.llama2 import Llama2Tokenizer
        self.tokenizer = Llama2Tokenizer.from_pretrained(model)
        super().__init__(model=self.model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self,
                inputs: str,
                max_length: int = 2048,
                do_sample: bool = False,
                top_p: float = 0.9,
                temperature: float = 0.6,
                repetition_penalty: float = 1.,
                eos_token_id: int = 2,
                bos_token_id: int = 1,
                pad_token_id: int = 0,
                system: str = 'you are a helpful assistant!',
                history: List = [],
                **forward_params) -> Dict[str, Any]:
        inputs_dict = forward_params
        inputs_dict['text'] = inputs
        inputs_dict['max_length'] = max_length
        inputs_dict['do_sample'] = do_sample
        inputs_dict['top_p'] = top_p
        inputs_dict['temperature'] = temperature
        inputs_dict['repetition_penalty'] = repetition_penalty
        inputs_dict['eos_token_id'] = eos_token_id
        inputs_dict['bos_token_id'] = bos_token_id
        inputs_dict['pad_token_id'] = pad_token_id
        inputs_dict['system'] = system
        inputs_dict['history'] = history
        output = self.model.chat(inputs_dict, self.tokenizer)
        return output

    # format the outputs from pipeline
    def postprocess(self, input, **kwargs) -> Dict[str, Any]:
        return input
