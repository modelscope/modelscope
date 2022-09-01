# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from io import BytesIO
from typing import Any, Dict, List, Union

import torch
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.pipelines.base import Input
from modelscope.preprocessors.image import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModeKeys, ModelFile, Tasks
from .base import Preprocessor
from .builder import PREPROCESSORS
from .ofa import *  # noqa
from .ofa.utils.collate import collate_fn

__all__ = [
    'OfaPreprocessor',
    'MPlugPreprocessor',
]


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.ofa_tasks_preprocessor)
class OfaPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode=ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
            mode: preprocessor mode (model mode)
        """
        super().__init__(*args, **kwargs)
        preprocess_mapping = {
            Tasks.image_captioning: OfaImageCaptioningPreprocessor,
            Tasks.visual_grounding: OfaVisualGroundingPreprocessor,
            Tasks.visual_question_answering:
            OfaVisualQuestionAnsweringPreprocessor,
            Tasks.visual_entailment: OfaVisualEntailmentPreprocessor,
            Tasks.image_classification: OfaImageClassificationPreprocessor,
            Tasks.text_classification: OfaTextClassificationPreprocessor,
            Tasks.summarization: OfaSummarizationPreprocessor,
            Tasks.text_to_image_synthesis: OfaTextToImageSynthesisPreprocessor
        }
        input_key_mapping = {
            Tasks.image_captioning: ['image'],
            Tasks.image_classification: ['image'],
            Tasks.summarization: ['text'],
            Tasks.text_classification: ['text', 'text2'],
            Tasks.visual_grounding: ['image', 'text'],
            Tasks.visual_question_answering: ['image', 'text'],
            Tasks.visual_entailment: ['image', 'text', 'text2'],
            Tasks.text_to_image_synthesis: ['text']
        }
        model_dir = model_dir if osp.exists(model_dir) else snapshot_download(
            model_dir)
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.preprocess = preprocess_mapping[self.cfg.task](
            cfg=self.cfg, model_dir=model_dir, mode=mode)
        self.keys = input_key_mapping[self.cfg.task]
        self.tokenizer = self.preprocess.tokenizer

    # just for modelscope demo
    def _build_dict(self, input: Union[Input, List[Input]]) -> Dict[str, Any]:
        data = dict()
        if not isinstance(input, tuple) and not isinstance(input, list):
            input = (input, )
        for key, item in zip(self.keys, input):
            data[key] = item
        return data

    def _compatible_with_pretrain(self, data):
        if 'image' in data and self.cfg.model.get('type', None) == 'ofa':
            image = load_image(data['image'])
            img_buffer = BytesIO()
            image.save(img_buffer, format='JPEG')
            data['image'] = Image.open(img_buffer)
        return data

    def __call__(self, input: Union[str, tuple, Dict[str, Any]], *args,
                 **kwargs) -> Dict[str, Any]:
        if isinstance(input, dict):
            data = input
        else:
            data = self._build_dict(input)
        data = self._compatible_with_pretrain(data)
        sample = self.preprocess(data)
        str_data = dict()
        for k, v in data.items():
            str_data[k] = str(v)
        sample['sample'] = str_data
        if kwargs.get('no_collate', None):
            return sample
        else:
            return collate_fn([sample],
                              pad_idx=self.tokenizer.pad_token_id,
                              eos_idx=self.tokenizer.eos_token_id)


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.mplug_tasks_preprocessor)
class MPlugPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir

        self._tokenizer = None
        self._patch_resize_transform = None

    @property
    def tokenizer(self):
        from transformers import BertTokenizer

        if self._tokenizer is None:
            self._tokenizer = BertTokenizer.from_pretrained(self.model_dir)
        return self._tokenizer

    @property
    def patch_resize_transform(self):
        if self._patch_resize_transform is None:
            from torchvision import transforms
            from modelscope.models.multi_modal.mplug import CONFIG_NAME, MPlugConfig

            config = MPlugConfig.from_yaml_file(
                osp.join(self.model_dir, CONFIG_NAME))

            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)

            self._patch_resize_transform = transforms.Compose([
                transforms.Resize((config.image_res, config.image_res),
                                  interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        return self._patch_resize_transform

    def __call__(self, *args, **kwargs):
        call_mapping = {
            Tasks.visual_question_answering: self.vqa_call,
            Tasks.image_captioning: self.caption_call
        }

        self.cfg = Config.from_file(
            osp.join(self.model_dir, ModelFile.CONFIGURATION))
        return call_mapping[self.cfg.task](*args, **kwargs)

    def vqa_call(self, data: Union[tuple, Dict[str, Any]]) -> Dict[str, Any]:
        image: Image.Image = data[0] if isinstance(data,
                                                   tuple) else data['image']
        question: str = data[1] if isinstance(data,
                                              tuple) else data['question']
        image = image.convert('RGB')
        image = self.patch_resize_transform(image)
        image = torch.stack([image], dim=0)
        question = self.tokenizer([question.lower()],
                                  padding='longest',
                                  return_tensors='pt')

        return {'image': image, 'question': question, 'train': False}

    def caption_call(
            self, data: Union[Image.Image, tuple,
                              Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(data, Image.Image):
            image = data
        elif isinstance(data, tuple):
            image = data[0]
        else:
            image = data['image']
        image = image.convert('RGB')
        image = self.patch_resize_transform(image)
        image = torch.stack([image], dim=0)
        question = self.tokenizer('', return_tensors='pt')

        return {'image': image, 'question': question, 'train': False}
