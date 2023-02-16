# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch

from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaSummarizationPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for summarization tasks.
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
        super(OfaSummarizationPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.mode == ModeKeys.TRAIN:
            return self._build_train_sample(data)
        else:
            return self._build_infer_sample(data)

    def _build_train_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Building training samples.

        step 1. Preprocess the data using the logic of `_build_infer_sample`
            and make sure the label data in the result.
        step 2. Preprocess the label data. Contains:
            - Get the lower case of label, and using `pre_caption` function
            to do the str preprocessing as new input label.
            - Tokenize the new input label as `target` for model input.
            - Add noise to the `target`
            - Calculate the `prev_output_tokens` from noise `target` for model input.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `image`, `prompt` and
                `label`, `image` refers the image input data, `prompt` refers the text input data
                and the `label` is the supervised data for training.
        Return:
            A dict object, contains source, image, mask, label, target tokens,
            and previous output tokens data.
        """
        sample = self._build_infer_sample(data)
        target_str = sample['label'].lower()
        target = super().pre_caption(target_str, max_words=self.max_tgt_length)
        target = target.replace('[unk]', 'unk').replace('<unk>', 'unk')
        sample['target'] = self.tokenize_text(target, add_bos=False)
        noise_target_item = self.add_noise_to_tgt(
            sample['target'][:-1].clone())
        sample['prev_output_tokens'] = torch.cat(
            [self.bos_item, noise_target_item])
        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Building inference samples.

        step 1. Preprocessing the input text via `pre_cation` function, see more
            details from the doc of `pre_cation`.
        step 2. Uniform the unknown token, such as `<unk>` -> `unk` and `<unk>` -> `unk`.
        step 3. Get the prompt from input, concatenate with the input text, as new input.
        step 4. Tokenize the input text and generate the decoder prompt.
        step 5. Determine Whether or not to add labels to the sample.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `image` and `prompt`,
                the former refers the image input data, and the later refers the text input data.
        Return:
            A dict object, contains text, decoder prompt and label data.
        """
        source = super().pre_caption(
            data[self.column_map['text']], max_words=self.max_src_length)
        source = source.replace('[unk]', 'unk').replace('<unk>', 'unk')
        prompt = self.cfg.model.get(
            'prompt', ' " {} " Summarize the article with a title: ')
        text = prompt.format(source)
        inputs = self.tokenize_text(text)
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
        if 'summary' in self.column_map and self.column_map['summary'] in data:
            sample['label'] = data[self.column_map['summary']]
        return sample

    def add_noise_to_tgt(self, target):
        r"""
        Add noise token to the target sentence.

        step 1. Sampling from uniform distribution to randomly select the
            noise indices.
        step 2. Sampling from normal distribution as noise token to replace
            the relative token in the target.

        Args:
            target: A sequence of tokens.
        Returns:
            A sequence of tokens.
        """
        noise_indices = torch.FloatTensor(
            target.size(0)).uniform_() < self.cfg.model.get(
                'noise_ratio', 0.0)
        target[noise_indices] = torch.randint(
            4,
            len(self.src_dict) - self.cfg.model.get('num_codes', 8192)
            - self.cfg.model.get('num_bins', 1000),
            size=(noise_indices.sum(), ))
        return target
