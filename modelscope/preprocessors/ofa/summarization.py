# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaSummarizationPreprocessor(OfaBasePreprocessor):

    def __init__(self,
                 cfg,
                 model_dir,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path,
            mode: preprocessor mode (model mode)
        """
        super(OfaSummarizationPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        source = super().pre_caption(
            data['text'], max_words=self.max_src_length)
        source = source.strip()[:self.max_src_length]
        source = source.replace('[unk]', 'unk').replace('<unk>', 'unk')
        prompt = self.cfg.model.get(
            'prompt', ' " {} " Summarize the article with a title: ')
        text = prompt.format(source)
        inputs = self.get_inputs(text)
        if self.prompt_type == 'none':
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'prev_output':
            decoder_prompt = inputs[:-1]
        else:
            raise NotImplementedError
        sample = {
            'source': inputs,
            'decoder_prompt': decoder_prompt,
        }
        return sample
