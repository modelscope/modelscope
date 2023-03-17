# Copyright (c) 2022 Zhipu.AI
import copy
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from transformers import AutoTokenizer, AutoModel


@MODELS.register_module(Tasks.text_generation, module_name=Models.chatglm6b)
class ChatGLM6bForTextGeneration(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the chatglm6b from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        self.logger = get_logger()
        # loading tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        # loading model
        self.model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()


    def forward(self, input: Dict) -> Dict:
        return {OutputKeys.TEXT: self.chat(input)}

    def chat(self, input: Dict) -> Dict:
        text = input['text']
        history = input['history']
        response, history = self.model.chat(self.tokenizer, text, history)
        self.logger.info('Generation finished.')
        res = {'response': response, 'history': history}
        return {OutputKeys.TEXT: res}
    
    def quantize(self, bits: int):
        self.model = self.model.quantize(bits)
        return self