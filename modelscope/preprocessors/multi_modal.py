# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, List, Tuple, Union

import torch
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.pipelines.base import Input
from modelscope.preprocessors import load_image
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

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data

        Args:
            model_dir (str): model path
        """
        super().__init__(*args, **kwargs)
        preprocess_mapping = {
            Tasks.ofa_ocr_recognition: OfaOcrRecognitionPreprocessor,
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
            Tasks.ofa_ocr_recognition: ['image'],
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
        self.preprocess = preprocess_mapping[self.cfg.task](self.cfg,
                                                            model_dir)
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

    def __call__(self, input: Union[str, tuple, Dict[str, Any]], *args,
                 **kwargs) -> Dict[str, Any]:
        if isinstance(input, dict):
            data = input
        else:
            data = self._build_dict(input)
        sample = self.preprocess(data)
        str_data = dict()
        for k, v in data.items():
            str_data[k] = str(v)
        sample['sample'] = str_data
        return collate_fn([sample],
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id)


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.mplug_tasks_preprocessor)
class MPlugPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 tokenizer_max_length: int = 25,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.mode = mode
        self.tokenizer_max_length = tokenizer_max_length

        self._tokenizer = None
        self._patch_resize_transform = None
        self._image_map = {}

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

    def image_open(self, path: str) -> Tuple[Image.Image, int]:
        if path not in self._image_map:
            index = len(self._image_map)
            self._image_map[path] = (load_image(path), index)
        return self._image_map[path]

    def __call__(
            self, data: Union[Image.Image, tuple,
                              Dict[str, Any]]) -> Dict[str, Any]:
        self.cfg = Config.from_file(
            osp.join(self.model_dir, ModelFile.CONFIGURATION))

        if isinstance(data, (Image.Image, str)):
            image = data
        elif isinstance(data, tuple):
            image = data[0]
        else:
            image = data['image']
        index = 0
        if isinstance(image, str):
            image, index = self.image_open(image)
        image = image.convert('RGB')
        image = self.patch_resize_transform(image)
        question = '' if self.cfg.task == Tasks.image_captioning \
            else data[1 if isinstance(data, tuple)
                      else ('text' if 'text' in data else 'question')]
        question = self.tokenizer(
            question.lower(),
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors='pt')

        if self.mode == ModeKeys.INFERENCE:
            image = torch.stack([image], dim=0)
            return {'image': image, 'question': question}
        else:
            answer = data['answer']
            answer = self.tokenizer(
                answer,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors='pt')
            output = {
                'image': image,
                'question_input_ids': question.input_ids.squeeze(),
                'question_attention_mask': question.attention_mask.squeeze(),
                'answer_input_ids': answer.input_ids.squeeze(),
                'answer_attention_mask': answer.attention_mask.squeeze(),
            }
            if self.cfg.task == Tasks.image_text_retrieval:
                output['index'] = index
            return output
