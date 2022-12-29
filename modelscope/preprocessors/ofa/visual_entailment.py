# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaVisualEntailmentPreprocessor(OfaBasePreprocessor):
    r"""
    OFA preprocessor for visual entailment tasks.
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
        super(OfaVisualEntailmentPreprocessor,
              self).__init__(cfg, model_dir, mode, *args, **kwargs)
        # Initialize transform
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize(
                (self.patch_image_size, self.patch_image_size),
                interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

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
        step 2. Preprocess the label data to generate the `target` and
        `prev_output_tokens`.
            - tokenize the label data.
            - calculate the target item.
                1) if `promp_type` is `None`, using tokenized label data.
                2) if `promp_type` is `src`, concatenating the `source` data
                and tokenized label data.
                3) if `promp_type` is `prev_output`, concatenating the `source`
                data without eos token and tokenized label data
        step 3. Add constraint mask

      Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `text`
                `text2` and `label` are optional.
        Return:
            A dict object, contains source text input, patch images, patch masks
            with `Tensor([True])` value, decoder prompt, label, target, previous
            output tokens and constraint mask.
        """
        sample = self._build_infer_sample(data)
        target = ' {}'.format(sample['label'])
        sample['ref_dict'] = {sample['label']: 1.0}
        tgt_item = self.tokenize_text(target, add_bos=False, add_eos=False)

        if self.prompt_type == 'none':
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        elif self.prompt_type == 'src':
            prev_output_item = torch.cat([sample['source'], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        elif self.prompt_type == 'prev_output':
            prev_output_item = torch.cat([sample['source'][:-1], tgt_item])
            target_item = torch.cat([prev_output_item[1:], self.eos_item])
        else:
            raise NotImplementedError

        target_item[:-len(tgt_item) - 1] = self.tokenizer.pad_token_id
        sample['target'] = target_item
        sample['prev_output_tokens'] = prev_output_item

        if self.constraint_trie is not None:
            constraint_mask = torch.zeros(
                (len(target_item), len(self.tgt_dict))).bool()
            start_idx = len(target_item) - len(tgt_item) - 1
            for i in range(
                    len(target_item) - len(tgt_item) - 1, len(target_item)):
                constraint_prefix_token = [
                    self.tgt_dict.bos()
                ] + target_item[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(
                    constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            sample['constraint_mask'] = constraint_mask

        return sample

    def _build_infer_sample(self, data: Dict[str, Any]) -> Dict[str, Any]:
        r"""
        Building inference samples.

        step 1. Preprocessing the image as model's image input.
            - get the pillow image input from `data`
            - do some transforms to the pillow image, such as resize, normalize etc.
        step 2. Building the instruction as model's source text input.
            - use text input to build instruction. so far, we support two kind of
            input form, we will take different examples to both of them to explain
            how to use them.
                1) only `text` input in data. this setting can solve the tasks which
                judge whether or not the input `text` describe the input image.
                2) both `text` and `text2` input in data. this setting can solve the
                tasks which judge whether or not the `text` together with input image
                can imply the `text2`
            - tokenize the instruction above.
        step 3. Calculate the decoder prompt input.
        step 4. Whether or not to add label data.

        Args:
            data (`Dict[str, Any]`): Input data, should contains the key of `text`
                `text2` and `label` are optional.
        Return:
            A dict object, contains source text input, patch images, patch masks
            with `Tensor([True])` value, decoder prompt and label.
        """
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        if 'text2' not in data:
            hypothesis = self.pre_caption(data[self.column_map['text']],
                                          self.max_src_length)
            prompt = self.cfg.model.get('prompt',
                                        ' does the image describe " {} "?')
            text = prompt.format(hypothesis)
        else:
            assert 'text' in data, f'text must be in the input {data.keys()}'
            caption = self.pre_caption(data[self.column_map['text2']],
                                       self.max_src_length)
            hypothesis = self.pre_caption(data[self.column_map['text']],
                                          self.max_src_length)
            prompt = self.cfg.model.get(
                'prompt', ' can image and text1 " {} " imply text2 " {} "?')
            text = prompt.format(caption, hypothesis)
        inputs = self.tokenize_text(text)
        if self.prompt_type == 'none':
            prefix_token = []
            decoder_prompt = self.bos_item
        elif self.prompt_type == 'prev_output':
            prefix_token = inputs[:-1]  # remove eos
            decoder_prompt = inputs[:-1]
        else:
            raise NotImplementedError
        sample = {
            'source': inputs,
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'prefix_token': prefix_token,
            'decoder_prompt': decoder_prompt,
        }
        if 'relation' in self.column_map and self.column_map[
                'relation'] in data:
            sample['label'] = data[self.column_map['relation']]
        return sample
