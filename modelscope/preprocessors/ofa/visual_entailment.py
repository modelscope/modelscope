# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Any, Dict

import torch
from PIL import Image
from torchvision import transforms

from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import ModeKeys
from .base import OfaBasePreprocessor


class OfaVisualEntailmentPreprocessor(OfaBasePreprocessor):

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

        target_item[:-len(tgt_item) - 1] = self.tgt_dict.pad()
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
        image = self.get_img_pil(data[self.column_map['image']])
        patch_image = self.patch_resize_transform(image)
        if 'text2' not in data:
            hypothesis = self.pre_caption(data['text'], self.max_src_length)
            prompt = self.cfg.model.get('prompt',
                                        ' does the image describe " {} "?')
            text = prompt.format(hypothesis)
        else:
            assert 'text' in data, f'text must be in the input {data.keys()}'
            caption = self.pre_caption(data['text2'], self.max_src_length)
            hypothesis = self.pre_caption(data['text'], self.max_src_length)
            prompt = self.cfg.model.get(
                'prompt', ' can image and text1 " {} " imply text2 " {} "?')
            text = prompt.format(caption, hypothesis)
        inputs = self.tokenize_text(text)
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
            'patch_image': patch_image,
            'patch_mask': torch.tensor([True]),
            'decoder_prompt': decoder_prompt,
        }
        if 'relation' in self.column_map and self.column_map[
                'relation'] in data:
            sample['label'] = data[self.column_map['relation']]
        return sample
