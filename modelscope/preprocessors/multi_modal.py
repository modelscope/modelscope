# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple, Union

import decord
import json
import numpy as np
import torch
from PIL import Image
from timm.data import create_transform
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Preprocessors
from modelscope.pipelines.base import Input
from modelscope.pipelines.cv.cmdssl_video_embedding_pipeline import (
    VCenterCrop, VCompose, VNormalize, VRescale, VToTensor)
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
    'DiffusionImageGenerationPreprocessor', 'OfaPreprocessor',
    'MPlugPreprocessor', 'HiTeAPreprocessor', 'MplugOwlPreprocessor'
]


@PREPROCESSORS.register_module(
    Fields.multi_modal,
    module_name=Preprocessors.diffusion_image_generation_preprocessor)
class DiffusionImageGenerationPreprocessor(Preprocessor):
    """ Preprocessor the data with the combination of image and text.
        Args:
            data: process the value as an image for keys ending with 'FILE'
                or existing in preprocessor_image_keys and pass-through the values of other keys.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.preprocessor_resolution = kwargs.pop('resolution', 512)
        self.preprocessor_mean = kwargs.pop('mean', [0.5])
        self.preprocessor_std = kwargs.pop('std', [0.5])
        self.preprocessor_image_keys = set(kwargs.pop('image_keys', []))
        self.center_crop = kwargs.pop('center_crop', True)

        self.transform_input = transforms.Compose([
            transforms.Resize(
                self.preprocessor_resolution,
                interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.preprocessor_resolution)
            if self.center_crop else transforms.RandomCrop(
                self.preprocessor_resolution),
            transforms.ToTensor(),
            transforms.Normalize(self.preprocessor_mean,
                                 self.preprocessor_std),
        ])

    def __call__(self, data) -> Dict[str, Any]:
        results = {}
        for key, value in data.items():
            if key.endswith(':FILE') or key in self.preprocessor_image_keys:
                image = load_image(value)
                img = self.transform_input(image)
                results[key.replace(':FILE', '').lower()] = img
            else:
                results[key.lower()] = value if value else ''
        return results


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
            Tasks.auto_speech_recognition: OfaASRPreprocessor,
            Tasks.sudoku: OfaSudokuPreprocessor,
            Tasks.text2sql: OfaTextToSqlPreprocessor
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


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.vldoc_preprocessor)
class VLDocPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        """Preprocess data for the model `VLDocForDocVLEmbedding`.

        Args:
            model_dir (str): model path in model hub.
            mode (str): model mode, in ('train', 'eval', 'inference').
        """
        super().__init__(*args, **kwargs)

        self.model_dir = model_dir
        self.mode = mode

        model_cfg_path = osp.join(model_dir, 'config.json')
        with open(model_cfg_path, 'r', encoding='utf-8') as f:
            model_cfg = json.load(f)

        from modelscope.models.multi_modal.vldoc.tokenization import VLDocXLMTokenizer
        tokenizer_path = osp.join(model_dir, ModelFile.TOKENIZER_FOLDER)
        self.tokenizer = VLDocXLMTokenizer.from_pretrained(tokenizer_path)

        from modelscope.models.multi_modal.vldoc.processing import Processor, ImageProcessor
        self.img_proc = ImageProcessor(
            do_preprocess=True,
            do_resize=True,
            image_size={
                'height': model_cfg['image_size'][0],
                'width': model_cfg['image_size'][1],
            },
            do_normalize=True,
            apply_ocr=False)
        self.proc = Processor(
            max_seq_length=model_cfg['max_seq_length'],
            max_block_num=model_cfg['max_block_num'],
            img_processor=self.img_proc,
            tokenizer=self.tokenizer,
            width=model_cfg['image_size'][1],
            height=model_cfg['image_size'][0],
        )

    def __call__(self, input: Dict[str, Any], *args,
                 **kwargs) -> Dict[str, Any]:
        """
        Args:
            input: {
                'images': ['img_path1', 'img_path2', ...],
                'ocr_info_paths': ['json_path1', 'json_path2', ...]
            }
        Return:
            encodings: Dict[str, Tensor]
        """

        ocr_infos = []
        for one_ocr_info_path in input['ocr_info_paths']:
            with open(one_ocr_info_path, 'r') as f:
                ocr_info = json.load(f)
                ocr_info = ocr_info['form']
                ocr_infos.append(ocr_info)

        proc_input = {'images': input['images'], 'ocr_infos': ocr_infos}
        encodings = self.proc(**proc_input)

        return encodings


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.hitea_tasks_preprocessor)
class HiTeAPreprocessor(Preprocessor):

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
        self._num_frames = None
        self._video_map = {}

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
            from modelscope.models.multi_modal.mplug import CONFIG_NAME, HiTeAConfig

            config = HiTeAConfig.from_yaml_file(
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

    @property
    def num_frames(self):
        if self._num_frames is None:
            from torchvision import transforms
            from modelscope.models.multi_modal.mplug import CONFIG_NAME, HiTeAConfig

            config = HiTeAConfig.from_yaml_file(
                osp.join(self.model_dir, CONFIG_NAME))

            self._num_frames = config.num_frames
        return self._num_frames

    def video_open(self, path: str) -> Tuple[decord.VideoReader, int]:
        if path not in self._video_map:
            index = len(self._video_map)
            vr = decord.VideoReader(path, ctx=decord.cpu(0))
            self._video_map[path] = (vr, index)
        return self._video_map[path]

    def sample_frames(self, num_frames: int, vlen: int) -> List[int]:
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(
            start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))

        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
        return frame_indices

    def __call__(
        self, data: Union[decord.VideoReader, tuple,
                          Dict[str, Any]]) -> Dict[str, Any]:
        self.cfg = Config.from_file(
            osp.join(self.model_dir, ModelFile.CONFIGURATION))

        if isinstance(data, (decord.VideoReader, str)):
            video = data
        elif isinstance(data, tuple):
            video = data[0]
        else:
            video = data['video']
        index = 0
        if isinstance(video, str):
            video, index = self.video_open(video)
        frame_indices = self.sample_frames(self.num_frames, len(video))
        video.seek(0)
        video = torch.from_numpy(video.get_batch(frame_indices).asnumpy())
        video = [
            self.patch_resize_transform(Image.fromarray(f))
            for f in video.numpy()
        ]
        video = torch.stack(video, dim=0)
        question = '' if self.cfg.task == Tasks.video_captioning \
            else data[1 if isinstance(data, tuple)
                      else ('text' if 'text' in data else 'question')]
        question = self.tokenizer(
            question.lower(),
            padding='max_length',
            truncation=True,
            max_length=self.tokenizer_max_length,
            return_tensors='pt')

        if self.mode == ModeKeys.INFERENCE:
            video = torch.stack([video], dim=0)
            return {'video': video, 'question': question}
        else:
            answer = data['answer']
            answer = self.tokenizer(
                answer,
                padding='max_length',
                truncation=True,
                max_length=self.tokenizer_max_length,
                return_tensors='pt')
            output = {
                'video': video,
                'question_input_ids': question.input_ids.squeeze(),
                'question_attention_mask': question.attention_mask.squeeze(),
                'answer_input_ids': answer.input_ids.squeeze(),
                'answer_attention_mask': answer.attention_mask.squeeze(),
            }
            return output


@PREPROCESSORS.register_module(
    Fields.multi_modal, module_name=Preprocessors.mplug_owl_preprocessor)
class MplugOwlPreprocessor(Preprocessor):

    def __init__(self,
                 model_dir: str,
                 mode: str = ModeKeys.INFERENCE,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.model_dir = model_dir
        self.mode = mode

        self._tokenizer = None
        self._patch_resize_transform = None
        self.media_token = {'<|image|>': 65}
        self._image_map = {}

    @property
    def tokenizer(self):
        from modelscope.models.nlp.llama import LlamaTokenizer

        if self._tokenizer is None:
            self._tokenizer = LlamaTokenizer.from_pretrained(self.model_dir)
        return self._tokenizer

    @property
    def patch_resize_transform(self):
        if self._patch_resize_transform is None:
            from torchvision import transforms

            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)

            self._patch_resize_transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
        return self._patch_resize_transform

    def image_open(self, path: str) -> Tuple[Image.Image, int]:
        if path not in self._image_map:
            index = len(self._image_map)
            self._image_map[path] = (load_image(path), index)
        return self._image_map[path]

    def tokenize_text(self, text: str) -> List[int]:
        media_tokens = {
            k: -int(i + 1)
            for i, k in enumerate(self.media_token.keys())
        }
        media_lengths = self.media_token.copy()

        prompt_chunk = [self.tokenizer.bos_token_id]

        # Pure Text
        condition = [
            media_token not in text for media_token in media_tokens.keys()
        ]
        if all(condition):
            enc_chunk = prompt_chunk + \
                self.tokenizer(text, add_special_tokens=False)['input_ids']

        # Multi-Modal Text
        else:
            enc_chunk = prompt_chunk
            pattern = '|'.join(map(re.escape, list(media_tokens.keys())))
            chunk_strs = re.split(f'({pattern})', text)
            chunk_strs = [x for x in chunk_strs if len(x) > 0]
            for idx, chunk_str in enumerate(chunk_strs):
                if chunk_str in media_tokens:
                    enc_chunk += [media_tokens[chunk_str]] * \
                        media_lengths[chunk_str]
                else:
                    tmp_chunk = self.tokenizer(
                        chunk_str, add_special_tokens=False)['input_ids']
                    enc_chunk += tmp_chunk
        return enc_chunk

    def convert(self, messages: Dict[str, List[Dict]]) -> str:
        texts = []
        image = []
        messages = messages['messages']
        for turn in messages:
            if turn['role'] == 'system':
                role = ''
            elif turn['role'] == 'user':
                role = 'Human: '
            else:
                role = 'AI: '
            if isinstance(turn['content'], str):
                text = f"{role}{turn['content']}"
                texts.append(text)
            else:
                for t in turn['content']:
                    if isinstance(t, str):
                        text = f'{role}{t}'
                    else:
                        text = f'{role}<|image|>'
                        image.append(t['image'])
                    texts.append(text)
        texts = '\n'.join(texts)
        texts += '\nAI: '
        return image, texts

    def __call__(self, messages: Dict[str, Any],
                 **forward_params) -> Dict[str, Any]:
        """
        Args:
            messages: {[
                {'role': 'system', 'content': 'message1'},
                {'role': 'user', 'content': 'message2'},
                {'role': 'user', 'content': ['message2', {"image": 'image_path'}, 'message3', ...]},
            ]}
            The 'role' should be choose from ['system', 'user', 'assistant'].
            The 'content' can be either str or List[Union[str, Dict]]
        Return:
            output: Dict[str, Tensor]
        """
        output = {}
        images, text = self.convert(messages)

        if len(images) > 0:
            pixel_values = []
            for image in images:
                pixel_values.append(
                    self.patch_resize_transform(self.image_open(image)[0]))
                pixel_values = torch.stack(pixel_values, dim=0)
        else:
            pixel_values = None

        input_ids = self.tokenize_text(text)
        input_ids = torch.LongTensor([input_ids])

        output = {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            **forward_params
        }

        return output


@PREPROCESSORS.register_module(
    Fields.multi_modal,
    module_name=Preprocessors.image_captioning_clip_interrogator_preprocessor)
class ImageCaptioningClipInterrogatorPreprocessor(Preprocessor):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, data) -> Dict[str, Any]:
        image = load_image(data)
        data = np.array(image).transpose(2, 0, 1)
        return data
