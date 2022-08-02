# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict, List, Union

import torch
from PIL import Image

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.pipelines.base import Input
from modelscope.utils.config import Config
from modelscope.utils.constant import Fields, ModelFile, Tasks
from .base import Preprocessor
from .builder import PREPROCESSORS
from .ofa import *  # noqa
from .ofa.utils.collate import collate_fn

__all__ = [
    'OfaPreprocessor',
    'MPlugVisualQuestionAnsweringPreprocessor',
]


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.ofa_image_caption)
class OfaPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            model_dir (str): model path
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
            Tasks.summarization: OfaSummarizationPreprocessor
        }
        input_key_mapping = {
            Tasks.image_captioning: ['image'],
            Tasks.image_classification: ['image'],
            Tasks.summarization: ['text'],
            Tasks.text_classification: ['text', 'text2'],
            Tasks.visual_grounding: ['image', 'text'],
            Tasks.visual_question_answering: ['image', 'text'],
            Tasks.visual_entailment: ['image', 'text', 'text2'],
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
        sample['sample'] = data
        return collate_fn([sample],
                          pad_idx=self.tokenizer.pad_token_id,
                          eos_idx=self.tokenizer.eos_token_id)


@PREPROCESSORS.register_module(
    Fields.multi_modal,
    module_name=Preprocessors.mplug_visual_question_answering)
class MPlugVisualQuestionAnsweringPreprocessor(Preprocessor):

    def __init__(self, model_dir: str, *args, **kwargs):
        """preprocess the data via 'bert-base-uncased' tokenizer and configuration

        """
        from transformers import BertTokenizer
        from modelscope.models.multi_modal.mplug import CONFIG_NAME, VOCAB_NAME, MPlugConfig

        super().__init__(*args, **kwargs)

        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            osp.join(model_dir, VOCAB_NAME))

        # load configuration
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
