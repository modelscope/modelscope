# Copyright (c) Alibaba, Inc. and its affiliates.
import io
import os
import re
import string
from os import path as osp

import json
import numpy as np
import torch
import torchaudio
from PIL import Image

from modelscope.fileio import File
from modelscope.models.multi_modal.ofa import OFATokenizer, OFATokenizerZH
from modelscope.preprocessors.image import load_image
from modelscope.utils.trie import Trie
from .utils.audio_helper import (_get_kaldi_fbank, _get_torchaudio_fbank,
                                 convert_waveform)
from .utils.constant import OFA_TASK_KEY_MAPPING
from .utils.random_help import set_torch_seed


class OfaBasePreprocessor:
    r"""
    OFA base preprocessor for
    """

    def __init__(self, cfg, model_dir, mode, *args, **kwargs):
        """preprocess the data via the vocab.txt from the `model_dir` path

        Args:
            cfg(modelscope.utils.config.ConfigDict) : model config
            model_dir (str): model path
        """
        self.cfg = cfg
        self.mode = mode
        self.language = self.cfg.model.get('language', 'en')
        if os.path.exists(model_dir):
            model_dir = os.path.abspath(model_dir)
        if self.language == 'en':
            tokenizer = OFATokenizer.from_pretrained(model_dir)
        elif self.language in ['zh', 'cn']:
            tokenizer = OFATokenizerZH.from_pretrained(model_dir)
        else:
            raise NotImplementedError
        # there is some diff between here and our ofa code,
        # there will be no need to use param: use_bpe
        tokenizer.add_tokens(['<code_{}>'.format(i) for i in range(8192)])
        tokenizer.add_tokens(['<bin_{}>'.format(i) for i in range(1000)])
        if self.cfg.model.get('multimodal_type', 'default') == 'text2sql':
            tokenizer.add_tokens(['>=', '<='])
        self.tokenizer = tokenizer
        self.bos_item = torch.LongTensor([tokenizer.bos_token_id])
        self.pad_item = torch.LongTensor([tokenizer.pad_token_id])
        self.eos_item = torch.LongTensor([tokenizer.eos_token_id])
        self.tgt_dict = self.src_dict = {
            value: key
            for key, value in tokenizer.get_vocab().items()
        }
        self.max_src_length = cfg.model.get('max_src_length', 256)
        self.max_tgt_length = cfg.model.get('max_tgt_length', 256)
        self.max_image_size = cfg.model.get('max_image_size', 512)
        self.language = self.cfg.model.get('language', 'en')
        self.prompt_type = self.cfg.model.get('prompt_type', 'none')
        seed = self.cfg.model.get('seed', 7)
        np.random.seed(seed)
        set_torch_seed(seed)
        imagenet_default_mean_and_std = self.cfg.model.get(
            'imagenet_default_mean_and_std', False)
        if imagenet_default_mean_and_std:
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]
        else:
            self.mean = [0.5, 0.5, 0.5]
            self.std = [0.5, 0.5, 0.5]
        self.patch_image_size = self.cfg.model.get('patch_image_size', 480)
        self.column_map = {
            key: key
            for key in OFA_TASK_KEY_MAPPING[self.cfg.task]
        }
        if hasattr(self.cfg,
                   'dataset') and self.cfg.dataset.column_map is not None:
            for k, v in self.cfg.dataset.column_map.items():
                self.column_map[k] = v
        self.transtab = str.maketrans(
            {key: None
             for key in string.punctuation})
        self.constraint_trie = None
        if self.cfg.model.get('answer2label', None):
            ans2label_file = osp.join(model_dir, self.cfg.model.answer2label)
            with open(ans2label_file, 'r', encoding='utf-8') as reader:
                ans2label_dict = json.load(reader)
            self.ans2label = ans2label_dict
            self.label2ans = {v: k for k, v in self.ans2label.items()}
            self.constraint_trie = Trie(tokenizer.eos_token_id)
            for i, answer in enumerate(ans2label_dict.keys()):
                answer_item = self.tokenize_text(
                    ' ' + answer, add_bos=False, add_eos=False)
                self.constraint_trie.insert([tokenizer.bos_token_id]
                                            + answer_item.tolist()
                                            + [tokenizer.eos_token_id])

        self.train_audio_feature_transforms = None
        self.test_audio_feature_transforms = None

    def tokenize_text(self, text, add_bos=True, add_eos=True):
        r"""
        Using `OFATokenizer` to tokenize text input.

        Args:
            text (`str`): Input text.
            add_bos ('bool', **optional**, default to `True`)
                Whether or not to add beginning of sentence token in
                the front of sentence.
            add_eos ('bool', **optional**, default to `True`)
                Whether or not to add ending of sentence token in
                the end of sentence.
        Returns:
            A list of tokens with the max length of `max_src_length + 2`
        """
        if text is None:
            return None
        inputs = self.tokenizer(
            text,
            max_length=self.max_src_length,
            add_special_tokens=False,
            truncation=True,
            return_tensors='pt')['input_ids'].squeeze(0)
        if add_bos:
            inputs = torch.cat([self.bos_item, inputs])
        if add_eos:
            inputs = torch.cat([inputs, self.eos_item])
        return inputs

    @staticmethod
    def pre_caption(caption, max_words=None):
        r"""
        Preprocessing for text sentence.

        step 1. Get the lower case of input text.
        step 2. Remove the words within `,.!?*#:;~ ` in the beginning
            of the sentence.
        step 3. Replace the words within `-/` or pattern `\s{2,}` with word ` `
            and replace tag `<person>` with `person`.
        step 4. Remove the `\n` in the end of the sentence.
        step 5. Split the sentence with token ` `, If `max_words` is not None,
            make a length truncation.

        Args:
            caption (`str`): Input text.
            max_words (`int`, **optional**, default `None`):
                The max length of input text. If None, do nothing, else
                make a truncation.

        Returns:
            A sequence of `str`.
        """
        caption = caption.lower().lstrip(',.!?*#:;~').replace('-', ' ') \
            .replace('/', ' ').replace('<person>', 'person')

        caption = re.sub(
            r'\s{2,}',
            ' ',
            caption,
        )
        caption = caption.rstrip('\n')
        caption = caption.strip(' ')

        # truncate caption
        caption_words = caption.split(' ')
        if max_words is not None and len(caption_words) > max_words:
            caption = ' '.join(caption_words[:max_words])

        return caption

    @staticmethod
    def pre_question(question, max_ques_words):
        r"""
        Preprocessing for text sentence.
        Note that this function is very similar to `pre_caption`, should be merged in the future version.

        step 1. Get the lower case of input text.
        step 2. Remove the words within `,.!?*#:;~ ` in the beginning
            of the sentence.
        step 3. Replace the words within `-/` or pattern `\s{2,}` with word ` `.
        step 4. Remove the `\n` in the end of the sentence.
        step 5. Split the sentence with token ` `, If `max_words` is not None,
            make a length truncation.

        Args:
            question (`str`): Input text.
            max_ques_words (`int`, **optional**, default `None`):
                The max length of input text. If None, do nothing, else
                make a truncation.

        Returns:
            A sequence of `str`.
        """
        question = question.lower().lstrip(',.!?*#:;~').replace('-',
                                                                ' ').replace(
                                                                    '/', ' ')

        question = re.sub(
            r'\s{2,}',
            ' ',
            question,
        )
        question = question.rstrip('\n')
        question = question.strip(' ')

        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])

        return question

    def add_constraint_mask(self, sample):
        r"""
        Add constraint mask.
        """
        target_itm = sample['target']
        len_label_itm = target_itm.ne(self.pad_item).sum(dim=0).item()
        if self.constraint_trie:
            constraint_mask = torch.zeros(
                (len(target_itm), len(self.tgt_dict))).bool()
            start_idx = len(target_itm) - len_label_itm
            for i in range(start_idx, len(target_itm)):
                constraint_prefix_token = self.bos_item.tolist(
                ) + target_itm[start_idx:i].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(
                    constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            sample['constraint_mask'] = constraint_mask

    def get_img_pil(self, path_or_url_or_pil):
        r"""
        Get the pillow image. If the input is not a pillow image ,it will load
        image from a local path or an external url.

        Args:
            path_or_url_or_pil (`Union[str, Image]`):
                Can be:
                    - A path or url reference to an image
                    - A pillow image.
        Returns:
            A pillow image.
        """
        image = path_or_url_or_pil if isinstance(path_or_url_or_pil, Image.Image) \
            else load_image(path_or_url_or_pil)
        return image

    def get_audio_bytes(self, path_or_url):
        if isinstance(path_or_url, bytes):
            audio_bytes = io.BytesIO(path_or_url)
        elif isinstance(path_or_url, str):
            file_bytes = File.read(path_or_url)
            audio_bytes = io.BytesIO(file_bytes)
        else:
            raise TypeError(f'Unsupported input type: {type(path_or_url)}.')
        return audio_bytes

    def prepare_fbank(self,
                      waveform,
                      sample_rate,
                      speed,
                      target_sample_rate=16000,
                      is_train=False):
        waveform, sample_rate = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(target_sample_rate)]])
        _waveform, _ = convert_waveform(
            waveform, sample_rate, to_mono=True, normalize_volume=True)
        # Kaldi compliance: 16-bit signed integers
        _waveform = _waveform * (2**15)
        _waveform = _waveform.numpy()
        fbank = _get_kaldi_fbank(_waveform, sample_rate, 80)
        if fbank is None:
            fbank = _get_torchaudio_fbank(_waveform, sample_rate, 80)
        if fbank is None:
            raise ImportError(
                'Please install pyKaldi or torchaudio to enable fbank feature extraction'
            )
        if is_train and self.train_audio_feature_transforms is not None:
            fbank = self.train_audio_feature_transforms(fbank)
        elif ~is_train and self.test_audio_feature_transforms(
                fbank) is not None:
            fbank = self.test_audio_feature_transforms(fbank)

        fbank = torch.from_numpy(fbank).float()
        fbank = self.pack_frames(fbank)
        return fbank

    def pack_frames(self, feature: torch.Tensor):
        if self.cfg.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.cfg.n_frames_per_step
        feature = feature[:self.cfg.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)
