# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaTextClassificationPreprocessor(OfaBasePreprocessor):

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
        super(OfaTextClassificationPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_instruction(self, data):
        text1 = ' '.join(
            data['text'].lower().strip().split()[:self.max_src_length])
        text2 = ' '.join(
            data['text2'].lower().strip().split()[:self.max_src_length])
        prompt = ' can text1 " {} " imply text2 " {} "?'
        text = prompt.format(text1, text2)
        instruction_itm = self.tokenize_text(text)
        return instruction_itm

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        instruction_itm = self._build_instruction(data)
        assert 'label' in data, 'there must has `label` column in train phase '
        label = data['label']
        if self.label2ans:
            label = self.label2ans[label]  # ans
        label_itm = self.tokenize_text(f' {label}', add_bos=False)
        if self.prompt_type == 'none':
            target_itm = label_itm
        elif self.prompt_type == 'prev_output':
            target_itm = torch.cat([instruction_itm[1:-1], label_itm])
        else:
            raise NotImplementedError
        prev_output_itm = torch.cat([self.bos_item, target_itm[:-1]])
        target_itm[:-len(label_itm)] = self.pad_item
        sample = {
            'source': instruction_itm,
            'target': target_itm,
            'prev_output_tokens': prev_output_itm,
        }
        self.add_constraint_mask(sample)
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        instruction_itm = self._build_instruction(data)
        if self.prompt_type == 'none':
            prefix_token = []
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'prev_output':
            prefix_token = instruction_itm[:-1]  # remove eos
            decoder_prompt = instruction_itm[:-1]
        else:
            raise NotImplementedError
        sample = {
            'source': instruction_itm,
            'prefix_token': prefix_token,
            'decoder_prompt': decoder_prompt,
        }
        if 'label' in data:
            sample['label'] = self.label2ans[data['label']]
        return sample
