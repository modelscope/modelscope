# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, Union

import torch
from PIL import Image
from torchvision import transforms

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.models.multi_modal.ofa import OFATokenizer
from modelscope.utils.constant import Fields
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS
from .image import load_image

__all__ = [
    'OfaImageCaptionPreprocessor',
    'MPlugVisualQuestionAnsweringPreprocessor',
]


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.ofa_image_caption)
class OfaImageCaptionPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        model_dir = model_dir if osp.exists(model_dir) else snapshot_download(
            model_dir)
        self.tokenizer = OFATokenizer.from_pretrained(model_dir)
        self.tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        self.tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(1000)])

        # Initialize transform
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        patch_image_size = 480
        self.patch_resize_transform = transforms.Compose([
            lambda image: image.convert('RGB'),
            transforms.Resize((patch_image_size, patch_image_size),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    @type_assert(object, (str, tuple, Image.Image))
    def __call__(self, data: Union[str, tuple]) -> Dict[str, Any]:
        if isinstance(data, Image.Image):
            patch_image = self.patch_resize_transform(data).unsqueeze(0)
        else:
            patch_image = self.patch_resize_transform(
                load_image(data)).unsqueeze(0)
        text = ' what does the image describe?'
        inputs = self.tokenizer([text], max_length=1024,
                                return_tensors='pt')['input_ids']
        sample = dict()
        sample['net_input'] = {
            'input_ids': inputs,
            'patch_images': patch_image,
            'patch_masks': torch.tensor([True])
        }
        return sample


@PREPROCESSORS.register_module(
    Fields.multi_modal,
    module_name=Preprocessors.mplug_visual_question_answering)
class MPlugVisualQuestionAnsweringPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via 'bert-base-uncased' tokenizer and configuration

        """
        super().__init__(*args, **kwargs)

        # tokenizer
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        # load configuration
        from sofa.models.mplug import CONFIG_NAME, MPlugConfig
        config = MPlugConfig.from_yaml_file(osp.join(model_dir, CONFIG_NAME))

        # Initialize transform
        from torchvision import transforms
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)

        self.patch_resize_transform = transforms.Compose([
            transforms.Resize((config.image_res, config.image_res),
                              interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        image, question = data['image'], data['question']
        image = Image.open(image).convert('RGB') if isinstance(image,
                                                               str) else image
        image = self.patch_resize_transform(image)
        image = torch.stack([image], dim=0)
        question = self.tokenizer([question.lower()],
                                  padding='longest',
                                  return_tensors='pt')

        return {'image': image, 'question': question, 'train': False}
