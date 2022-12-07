# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import json
import torch
from PIL import Image
from timm.data import create_transform
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.pipelines.base import Input
from modelscope.preprocessors import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import (Fields, Invoke, ModeKeys, ModelFile,
                                       Tasks)
from .base import Preprocessor
from .builder import PREPROCESSORS
from .ofa import *  # noqa
from .ofa.utils.collate import collate_fn
from .ofa.utils.constant import OFA_TASK_KEY_MAPPING

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
            Tasks.ocr_recognition: OfaOcrRecognitionPreprocessor,
            Tasks.image_captioning: OfaImageCaptioningPreprocessor,
            Tasks.visual_grounding: OfaVisualGroundingPreprocessor,
            Tasks.visual_question_answering:
            OfaVisualQuestionAnsweringPreprocessor,
            Tasks.visual_entailment: OfaVisualEntailmentPreprocessor,
            Tasks.image_classification: OfaImageClassificationPreprocessor,
            Tasks.text_classification: OfaTextClassificationPreprocessor,
            Tasks.text_summarization: OfaSummarizationPreprocessor,
            Tasks.text_to_image_synthesis: OfaTextToImageSynthesisPreprocessor,
            Tasks.auto_speech_recognition: OfaASRPreprocessor
        }
        model_dir = model_dir if osp.exists(model_dir) else snapshot_download(
            model_dir, user_agent={Invoke.KEY: Invoke.PREPROCESSOR})
        self.cfg = Config.from_file(
            osp.join(model_dir, ModelFile.CONFIGURATION))
        self.preprocess = preprocess_mapping[self.cfg.task](
            cfg=self.cfg, model_dir=model_dir, mode=mode)
        self.keys = OFA_TASK_KEY_MAPPING[self.cfg.task]
        self.tokenizer = self.preprocess.tokenizer
        if kwargs.get('no_collate', None):
            self.no_collate = True
        else:
            self.no_collate = False

    # just for modelscope demo
    def _build_dict(self, input: Union[Input, List[Input]]) -> Dict[str, Any]:
        data = dict()
        if not isinstance(input, tuple) and not isinstance(input, list):
            input = (input, )
        for key, item in zip(self.keys, input):
            data[key] = item
        return data

    def _ofa_input_compatibility_conversion(self, data):  # fake
        if 'image' in data and self.cfg.model.get('type', None) == 'ofa':
            if isinstance(data['image'], str):
                image = load_image(data['image'])
            else:
                image = data['image']
            if image.mode != 'RGB':
                image = image.convert('RGB')
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
        sample = self.preprocess(data)
        str_data = dict()
        for k, v in data.items():
            str_data[k] = str(v)
        sample['sample'] = str_data
        if self.no_collate:
            return sample
        else:
            return collate_fn([sample],
                              pad_idx=self.tokenizer.pad_token_id,
                              eos_idx=self.tokenizer.eos_token_id)


def _convert_to_rgb(image):
    return image.convert('RGB')


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.clip_preprocessor)
class CLIPPreprocessor(Preprocessor):

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
        model_dir = model_dir if osp.exists(model_dir) else snapshot_download(
            model_dir, user_agent={Invoke.KEY: Invoke.PREPROCESSOR})
        self.mode = mode
        # text tokenizer
        from modelscope.models.multi_modal.clip.bert_tokenizer import FullTokenizer
        if 'tokenizer' in kwargs and isinstance(kwargs['tokenizer'],
                                                FullTokenizer):
            self.tokenizer = kwargs['tokenizer']
        else:
            vocab_file = f'{model_dir}/{ModelFile.VOCAB_FILE}'
            self.tokenizer = FullTokenizer(vocab_file=vocab_file)
        # image preprocessor
        if 'resolution' in kwargs and isinstance(kwargs['resolution'], int):
            self.image_resolution = kwargs['resolution']
        else:
            self.image_resolution = json.load(
                open(
                    '{}/vision_model_config.json'.format(model_dir),
                    encoding='utf-8'))['image_resolution']
        self.img_preprocess = self._build_image_transform()
        # key mapping
        # specify the input keys, compatible with training and inference whose key names may be different
        self.input_keys = {'img': 'img', 'text': 'text'}

    def _build_image_transform(self):

        if self.mode == ModeKeys.TRAIN:
            transform = create_transform(
                input_size=self.image_resolution,
                scale=(0.9, 1.0),
                is_training=True,
                color_jitter=None,
                auto_augment='original',
                interpolation='bicubic',
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            )
            transform = Compose(transform.transforms[:-3] + [_convert_to_rgb]
                                + transform.transforms[-3:])
        else:
            transform = Compose([
                Resize((self.image_resolution, self.image_resolution),
                       interpolation=Image.BICUBIC),
                _convert_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073),
                          (0.26862954, 0.26130258, 0.27577711)),
            ])
        return transform

    def tokenize(self,
                 texts: Union[str, List[str]],
                 context_length: int = 52) -> torch.LongTensor:
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all baseline models use 24 as the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
        """
        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            all_tokens.append(
                [self.tokenizer.vocab['[CLS]']]
                + self.tokenizer.convert_tokens_to_ids(
                    self.tokenizer.tokenize(text))[:context_length - 2]
                + [self.tokenizer.vocab['[SEP]']])

        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            assert len(tokens) <= context_length
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def set_input_img_key(self, new_key: str):
        self.input_keys['img'] = new_key

    def set_input_text_key(self, new_key: str):
        self.input_keys['text'] = new_key

    def __call__(self, input: Union[str, tuple, Dict[str, Any]], *args,
                 **kwargs) -> Dict[str, Any]:
        output = {}
        # preprocess the image input
        input_img_key = self.input_keys['img']
        if input_img_key in input and input[input_img_key] is not None:
            image_input = input[input_img_key]

            # single image input
            if isinstance(image_input, Image.Image):
                image_tensor = self.img_preprocess(image_input).unsqueeze(0)
            # multi images input
            elif isinstance(image_input, list):
                if all([isinstance(elem, Image.Image)
                        for elem in image_input]):
                    image_tensor = torch.stack(
                        [self.img_preprocess(elem)
                         for elem in image_input],  # noqa
                        dim=0)  # noqa
                else:
                    unsupported_elem_type = [
                        type(elem) for elem in image_input
                        if not isinstance(elem, Image.Image)
                    ][0]
                    raise TypeError(
                        f'img should be PIL.Image or List[PIL.Image], \
                            but got a List containing one {unsupported_elem_type}'
                    )
            # others
            else:
                raise TypeError(
                    f'img should be PIL.Image or List[PIL.Image], but got {type(image_input)}'
                )
            output['img'] = image_tensor

        # preprocess the text input
        input_text_key = self.input_keys['text']
        if input_text_key in input and input[input_text_key] is not None:
            text_input = input[input_text_key]

            # single text input
            if isinstance(text_input, str):
                text_tensor = self.tokenize(text_input)
            # multi texts input
            elif isinstance(text_input, list):
                if all([isinstance(elem, str) for elem in text_input]):
                    text_tensor = self.tokenize(text_input)
                else:
                    unsupported_elem_type = [
                        type(elem) for elem in text_input
                        if not isinstance(elem, str)
                    ][0]
                    raise TypeError(
                        f'text should be str or List[str], but got a List containing one {unsupported_elem_type}'
                    )
            # others
            else:
                raise TypeError(
                    f'text should be str or List[str], but got {type(text_input)}'
                )
            output['text'] = text_tensor

        return output


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
