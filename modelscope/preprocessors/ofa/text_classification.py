# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from .base import OfaBasePreprocessor


class OfaTextClassificationPreprocessor(OfaBasePreprocessor):

    def __init__(self, cfg, model_dir):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path
        """
        super(OfaTextClassificationPreprocessor, self).__init__(cfg, model_dir)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text1 = ' '.join(
            data['text'].lower().strip().split()[:self.max_src_length])
        text2 = ' '.join(
            data['text2'].lower().strip().split()[:self.max_src_length])
        prompt = ' can text1 " {} " imply text2 " {} "?'
        text = prompt.format(text1, text2)
        inputs = self.get_inputs(text)
        if self.prompt_type == 'none':
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'src':
            decoder_prompt = inputs
        elif self.prompt_type == 'prev_output':
            decoder_prompt = inputs[:-1]
        else:
            raise NotImplementedError
        sample = {
            'source': inputs,
            'decoder_prompt': decoder_prompt,
        }
        return sample
