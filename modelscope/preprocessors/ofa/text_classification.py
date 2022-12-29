# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaTextClassificationPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for text classification tasks.
    """

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
        r"""
        Building text classification task's instruction.

        The `data` should contains key `text` and `text2`, and the final instruction
        is like  ` can text1 " {} " imply text2 " {} "?`, the first `{}` refer to
        the value of `text` and the latter refer to `text2`

        step 1. Preprocess for input text `text` and `text2` in `data`.
            - Do lower, stripe and restrict the maximum length as `max_src_length`.
        step 2. Using instruction template to generate the final instruction.
        step 3. Tokenize the instruction as result.
        """
        text1 = ' '.join(
            data['text'].lower().strip().split()[:self.max_src_length])
        text2 = ' '.join(
            data['text2'].lower().strip().split()[:self.max_src_length])
        prompt = ' can text1 " {} " imply text2 " {} "?'
        text = prompt.format(text1, text2)
        instruction_itm = self.tokenize_text(text)
        return instruction_itm

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Building training samples.

        step 1. Building instruction for text classification using `_build_instruction`.
        step 2. If the `label` is not text, transfer it to text using `label2ans`.
        step 3. Tokenize the label data.
        step 4. Concatenate the instruction and label tokens as the target item.
            - padding the instruction tokens from target item as `target`.
            - remove the eos token from target item as `prev_output_tokens`.
        step 5. Add constraint mask.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `text`, `text2`
                and `label`, both of them refer to a text input, and the target of this job
                is to find whether or not `text` imply `text2`, the `label` is the supervised
                data for training.
        Return:
            A dict object, contains source text input, target tokens and previous output
            tokens and constraint mask.
        """
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
        r"""
        Building inference samples.

        step 1. Building instruction for text classification using `_build_instruction`.
        step 2. Whether or not to add `prefix_token`.
        step 3. Whether or not to add `label` data.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `text` and `text2`,
                both of them refer to a text input, and the target of this job is to find
                whether or not `text` imply `text2`.
        Return:
            A dict object, contains source text input, prefix tokens and label data.
        """
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
