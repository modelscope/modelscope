# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright (c) 2022 Zhipu.AI
from typing import Any, Dict, Union

import torch

from modelscope import Model, snapshot_download
from modelscope.metainfo import Pipelines, Preprocessors
from modelscope.models.nlp.llama2 import Llama2Tokenizer
from modelscope.pipelines.base import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.pipelines.nlp.text_generation_pipeline import \
    TextGenerationPipeline
from modelscope.preprocessors import Preprocessor
from modelscope.utils.constant import Fields, Tasks


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
            >>> ignore_file_pattern = [r'\\w+\\.safetensors'])
            >>> pipe = pipeline(task=Tasks.text_generation, model=model_dir, device_map='auto',
            >>> torch_dtype=torch.float16)
            >>> inputs="咖啡的作用是什么？"
            >>> result = pipe(inputs,max_length=200, do_sample=True, top_p=0.85,
            >>> temperature=1.0, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
            >>> print(result['text'])

            To view other examples plese check tests/pipelines/test_llama2_text_generation_pipeline.py.
        """
        self.model = Model.from_pretrained(
            model, device_map='auto', torch_dtype=torch.float16)
        self.tokenizer = Llama2Tokenizer.from_pretrained(model)
        super().__init__(model=self.model, **kwargs)

    def preprocess(self, inputs, **preprocess_params) -> Dict[str, Any]:
        return inputs

    def _sanitize_parameters(self, **pipeline_parameters):
        return {}, pipeline_parameters, {}

    def forward(self,
                inputs,
                max_length=2048,
                do_sample=True,
                top_p=0.85,
                temperature=1.0,
                repetition_penalty=1.,
                eos_token_id=2,
                bos_token_id=1,
                pad_token_id=0,
                **forward_params) -> Dict[str, Any]:
        output = {}
        inputs = self.tokenizer(inputs, return_tensors='pt')
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
