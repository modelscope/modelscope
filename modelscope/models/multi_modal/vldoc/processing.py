# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Processor class for GeoLayoutLM.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Union

import cv2
import numpy as np
import PIL
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import transforms

from modelscope.preprocessors.image import LoadImage


def custom_tokenize(tokenizer, text):
    toks = tokenizer.tokenize('pad ' + text)[1:]
    toks2 = toks[1:] if len(toks) > 0 and toks[0] == '▁' else toks
    return toks2


class ImageProcessor(object):
    r"""
    Construct a GeoLayoutLM image processor
    Args:
        do_preprocess (`bool`): whether to do preprocess to unify the image format,
            resize and convert to tensor.
        do_rescale: only works when we disable do_preprocess.
    """

    def __init__(self,
                 do_preprocess: bool = True,
                 do_resize: bool = False,
                 image_size: Dict[str, int] = None,
                 do_rescale: bool = False,
                 rescale_factor: float = 1. / 255,
                 do_normalize: bool = True,
                 image_mean: Union[float, Iterable[float]] = None,
                 image_std: Union[float, Iterable[float]] = None,
                 apply_ocr: bool = True,
                 **kwargs) -> None:
        self.do_preprocess = do_preprocess
        self.do_resize = do_resize
        self.size = image_size if image_size is not None else {
            'height': 768,
            'width': 768
        }
        self.do_rescale = do_rescale and (not do_preprocess)
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        image_mean = IMAGENET_DEFAULT_MEAN if image_mean is None else image_mean
        image_std = IMAGENET_DEFAULT_STD if image_std is None else image_std
        self.image_mean = (image_mean, image_mean, image_mean) if isinstance(
            image_mean, float) else image_mean
        self.image_std = (image_std, image_std, image_std) if isinstance(
            image_std, float) else image_std
        self.apply_ocr = apply_ocr
        self.kwargs = kwargs

        self.totensor = transforms.ToTensor()

    def preprocess(self, image: Union[np.ndarray, PIL.Image.Image]):
        """ unify the image format, resize and convert to tensor.
        """
        image = LoadImage.convert_to_ndarray(image)[:, :, ::-1]
        size_raw = image.shape[:2]
        if self.do_resize:
            image = cv2.resize(image,
                               (self.size['width'], self.size['height']))
        # convert to pytorch tensor
        image_pt = self.totensor(image)
        return image_pt, size_raw

    def __call__(self, images: Union[list, np.ndarray, PIL.Image.Image, str]):
        """
        Args:
            images: list of np.ndarrays, PIL images or image tensors.
        """
        if not isinstance(images, list):
            images = [images]
        sizes_raw = []
        if self.do_preprocess:
            for i in range(len(images)):
                images[i], size_raw = self.preprocess(images[i])
                sizes_raw.append(size_raw)
        images_pt = torch.stack(images, dim=0)  # [b, c, h, w]
        if self.do_rescale:
            images_pt = images_pt * self.rescale_factor
        if self.do_normalize:
            mu = torch.tensor(self.image_mean).view(1, 3, 1, 1)
            std = torch.tensor(self.image_std).view(1, 3, 1, 1)
            images_pt = (images_pt - mu) / (std + 1e-8)

        # TODO: apply OCR
        ocr_infos = None
        if self.apply_ocr:
            raise NotImplementedError('OCR service is not available yet!')
        if len(sizes_raw) == 0:
            sizes_raw = None
        data = {
            'images': images_pt,
            'ocr_infos': ocr_infos,
            'sizes_raw': sizes_raw
        }
        return data


class OCRUtils(object):

    def __init__(self):
        self.version = 'v0'

    def __call__(self, ocr_infos):
        """
        sort boxes, filtering or other preprocesses
        should return sorted ocr_infos
        """
        raise NotImplementedError


def bound_box(box, height, width):
    # box: [x_tl, y_tl, x_br, y_br] or ...
    assert len(box) == 4 or len(box) == 8
    for i in range(len(box)):
        if i & 1:
            box[i] = max(0, min(box[i], height))
        else:
            box[i] = max(0, min(box[i], width))
    return box


def bbox2pto4p(box2p):
    box4p = [
        box2p[0], box2p[1], box2p[2], box2p[1], box2p[2], box2p[3], box2p[0],
        box2p[3]
    ]
    return box4p


def bbox4pto2p(box4p):
    box2p = [
        min(box4p[0], box4p[2], box4p[4], box4p[6]),
        min(box4p[1], box4p[3], box4p[5], box4p[7]),
        max(box4p[0], box4p[2], box4p[4], box4p[6]),
        max(box4p[1], box4p[3], box4p[5], box4p[7]),
    ]
    return box2p


def stack_tensor_dict(tensor_dicts: List[Dict[str, torch.Tensor]]):
    one_dict = defaultdict(list)
    for td in tensor_dicts:
        for k, v in td.items():
            one_dict[k].append(v)
    res_dict = {}
    for k, v in one_dict.items():
        res_dict[k] = torch.stack(v, dim=0)
    return res_dict


class TextLayoutSerializer(object):

    def __init__(self,
                 max_seq_length: int,
                 max_block_num: int,
                 tokenizer,
                 width=768,
                 height=768,
                 use_roberta_tokenizer: bool = True,
                 ocr_utils: OCRUtils = None):
        self.version = 'v0'

        self.max_seq_length = max_seq_length
        self.max_block_num = max_block_num
        self.tokenizer = tokenizer
        self.width = width
        self.height = height
        self.use_roberta_tokenizer = use_roberta_tokenizer
        self.ocr_utils = ocr_utils

        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.bos_token_id
        self.sep_token_id = tokenizer.eos_token_id
        self.unk_token_id = tokenizer.unk_token_id
        self.cls_bbs_word = [0.0] * 8
        self.cls_bbs_line = [0] * 4

    def label2seq(self, ocr_info: list, label_info: list):
        raise NotImplementedError

    def serialize_single(
        self,
        ocr_info: list = None,
        input_ids: list = None,
        bbox_line: List[List] = None,
        bbox_word: List[List] = None,
        width: int = 768,
        height: int = 768,
    ):
        r"""
        Either ocr_info or (input_ids, bbox_line, bbox_word)
            should be provided.
        If (input_ids, bbox_line, bbox_word) is provided,
            convinient plug into the serialization (customization)
            is offered. The tokens must be organised by blocks and words.
        Else, ocr_info must be provided, to be parsed
            to sequences directly (the simplest way).
        Args:
            ocr_info: [
                {"text": "xx", "box": [a,b,c,d],
                 "words": [{"text": "x", "box": [e,f,g,h]}, ...]},
                ...
            ]
            bbox_line: the coordinate value should match the original image
                (i.e., not be normalized).
        """
        if input_ids is not None:
            assert len(input_ids) == len(bbox_line)
            assert len(input_ids) == len(bbox_word)
            input_ids, bbs_word, bbs_line, first_token_idxes, \
                line_rank_ids, line_rank_inner_ids, word_rank_ids = \
                self.halfseq2seq(input_ids, bbox_line, bbox_word, width, height)
        else:
            assert ocr_info is not None
            input_ids, bbs_word, bbs_line, first_token_idxes, \
                line_rank_ids, line_rank_inner_ids, word_rank_ids = \
                self.ocr_info2seq(ocr_info, width, height)

        token_seq = {}
        token_seq['input_ids'] = torch.ones(
            self.max_seq_length, dtype=torch.int64) * self.pad_token_id
        token_seq['attention_mask'] = torch.zeros(
            self.max_seq_length, dtype=torch.int64)
        token_seq['first_token_idxes'] = torch.zeros(
            self.max_block_num, dtype=torch.int64)
        token_seq['first_token_idxes_mask'] = torch.zeros(
            self.max_block_num, dtype=torch.int64)
        token_seq['bbox_4p_normalized'] = torch.zeros(
            self.max_seq_length, 8, dtype=torch.float32)
        token_seq['bbox'] = torch.zeros(
            self.max_seq_length, 4, dtype=torch.float32)
        token_seq['line_rank_id'] = torch.zeros(
            self.max_seq_length, dtype=torch.int64)  # start from 1
        token_seq['line_rank_inner_id'] = torch.ones(
            self.max_seq_length, dtype=torch.int64)  # 1 2 2 3
        token_seq['word_rank_id'] = torch.zeros(
            self.max_seq_length, dtype=torch.int64)  # start from 1

        # expand using cls and sep tokens
        sep_bbs_word = [width, height] * 4
        sep_bbs_line = [width, height] * 2
        input_ids = [self.cls_token_id] + input_ids + [self.sep_token_id]
        bbs_line = [self.cls_bbs_line] + bbs_line + [sep_bbs_line]
        bbs_word = [self.cls_bbs_word] + bbs_word + [sep_bbs_word]

        # assign
        len_tokens = len(input_ids)
        len_lines = len(first_token_idxes)
        token_seq['input_ids'][:len_tokens] = torch.tensor(input_ids)
        token_seq['attention_mask'][:len_tokens] = 1
        token_seq['first_token_idxes'][:len_lines] = torch.tensor(
            first_token_idxes)
        token_seq['first_token_idxes_mask'][:len_lines] = 1
        token_seq['line_rank_id'][1:len_tokens
                                  - 1] = torch.tensor(line_rank_ids)
        token_seq['line_rank_inner_id'][1:len_tokens - 1] = torch.tensor(
            line_rank_inner_ids)
        token_seq['line_rank_inner_id'] = token_seq[
            'line_rank_inner_id'] * token_seq['attention_mask']
        token_seq['word_rank_id'][1:len_tokens
                                  - 1] = torch.tensor(word_rank_ids)

        token_seq['bbox_4p_normalized'][:len_tokens, :] = torch.tensor(
            bbs_word)
        # word bbox normalization -> [0, 1]
        token_seq['bbox_4p_normalized'][:, [0, 2, 4, 6]] = \
            token_seq['bbox_4p_normalized'][:, [0, 2, 4, 6]] / width
        token_seq['bbox_4p_normalized'][:, [1, 3, 5, 7]] = \
            token_seq['bbox_4p_normalized'][:, [1, 3, 5, 7]] / height

        token_seq['bbox'][:len_tokens, :] = torch.tensor(bbs_line)
        # line bbox -> [0, 1000)
        token_seq['bbox'][:,
                          [0, 2]] = token_seq['bbox'][:, [0, 2]] / width * 1000
        token_seq['bbox'][:,
                          [1, 3]] = token_seq['bbox'][:,
                                                      [1, 3]] / height * 1000
        token_seq['bbox'] = token_seq['bbox'].long()

        return token_seq

    def ocr_info2seq(self, ocr_info: list, width: int, height: int):
        input_ids = []
        bbs_word = []
        bbs_line = []
        first_token_idxes = []
        line_rank_ids = []
        line_rank_inner_ids = []
        word_rank_ids = []

        early_stop = False
        for line_idx, line in enumerate(ocr_info):
            if line_idx == self.max_block_num:
                early_stop = True
            if early_stop:
                break
            lbox = line['box']
            lbox = bound_box(lbox, height, width)
            is_first_word = True
            for word_id, word_info in enumerate(line['words']):
                wtext = word_info['text']
                wbox = word_info['box']
                wbox = bound_box(wbox, height, width)
                wbox4p = bbox2pto4p(wbox)
                if self.use_roberta_tokenizer:
                    wtokens = custom_tokenize(self.tokenizer, wtext)
                else:
                    wtokens = self.tokenizer.tokenize(wtext)
                wtoken_ids = self.tokenizer.convert_tokens_to_ids(wtokens)
                if len(wtoken_ids) == 0:
                    wtoken_ids.append(self.unk_token_id)
                n_tokens = len(wtoken_ids)
                # reserve for cls and sep
                if len(input_ids) + n_tokens > self.max_seq_length - 2:
                    early_stop = True
                    break  # chunking early for long documents
                if is_first_word:
                    first_token_idxes.append(len(input_ids) + 1)
                input_ids.extend(wtoken_ids)
                bbs_word.extend([wbox4p] * n_tokens)
                bbs_line.extend([lbox] * n_tokens)
                word_rank_ids.extend([word_id + 1] * n_tokens)
                line_rank_ids.extend([line_idx + 1] * n_tokens)
                if is_first_word:
                    if len(line_rank_inner_ids
                           ) > 0 and line_rank_inner_ids[-1] == 2:
                        line_rank_inner_ids[-1] = 3
                    line_rank_inner_ids.extend([1] + (n_tokens - 1) * [2])
                    is_first_word = False
                else:
                    line_rank_inner_ids.extend(n_tokens * [2])
        if len(line_rank_inner_ids) > 0 and line_rank_inner_ids[-1] == 2:
            line_rank_inner_ids[-1] = 3

        return input_ids, bbs_word, bbs_line, first_token_idxes, line_rank_ids, \
            line_rank_inner_ids, word_rank_ids

    def halfseq2seq(self, input_ids: list, bbox_line: List[List],
                    bbox_word: List[List], width: int, height: int):
        """
        for convinient plug into the serialization, given the 3 customized sequences.
        They should not contain special tokens like [CLS] or [SEP].
        """
        bbs_word = []
        bbs_line = []
        first_token_idxes = []
        line_rank_ids = []
        line_rank_inner_ids = []
        word_rank_ids = []

        n_real_tokens = len(input_ids)
        lb_prev, wb_prev = None, None
        line_id = 0
        word_id = 1
        for i in range(n_real_tokens):
            lb_now = bbox_line[i]
            wb_now = bbox_word[i]
            line_start = lb_prev is None or lb_now != lb_prev
            word_start = wb_prev is None or wb_now != wb_prev
            lb_prev, wb_prev = lb_now, wb_now

            if len(lb_now) == 8:
                lb_now = bbox4pto2p(lb_now)
            assert len(lb_now) == 4
            lb_now = bound_box(lb_now, height, width)
            if len(wb_now) == 4:
                wb_now = bbox2pto4p(wb_now)
            assert len(wb_now) == 8
            wb_now = bound_box(wb_now, height, width)

            bbs_word.append(wb_now)
            bbs_line.append(lb_now)

            if word_start:
                word_id += 1
            if line_start:
                line_id += 1
                first_token_idxes.append(i + 1)
                if len(line_rank_inner_ids
                       ) > 0 and line_rank_inner_ids[-1] == 2:
                    line_rank_inner_ids[-1] = 3
                line_rank_inner_ids.append(1)
                word_id = 1
            else:
                line_rank_inner_ids.append(2)
            line_rank_ids.append(line_id)
            word_rank_ids.append(word_id)

        if len(line_rank_inner_ids) > 0 and line_rank_inner_ids[-1] == 2:
            line_rank_inner_ids[-1] = 3

        return input_ids, bbs_word, bbs_line, first_token_idxes, \
            line_rank_ids, line_rank_inner_ids, word_rank_ids

    def __call__(
        self,
        ocr_infos: List[List] = None,
        input_ids: list = None,
        bboxes_line: List[List] = None,
        bboxes_word: List[List] = None,
        sizes_raw: list = None,
        **kwargs,
    ):
        n_samples = len(ocr_infos) if ocr_infos is not None else len(input_ids)
        if sizes_raw is None:
            sizes_raw = [(self.height, self.width)] * n_samples
        seqs = []
        if input_ids is not None:
            assert len(input_ids) == len(bboxes_line)
            assert len(input_ids) == len(bboxes_word)
            for input_id, bbox_line, bbox_word, size_raw in zip(
                    input_ids, bboxes_line, bboxes_word, sizes_raw):
                height, width = size_raw
                token_seq = self.serialize_single(None, input_id, bbox_line,
                                                  bbox_word, width, height)
                seqs.append(token_seq)
        else:
            assert ocr_infos is not None, 'For serialization, ocr_infos must not be NoneType!'
            if self.ocr_utils is not None:
                ocr_infos = self.ocr_utils(ocr_infos)
            for ocr_info, size_raw in zip(ocr_infos, sizes_raw):
                height, width = size_raw
                token_seq = self.serialize_single(
                    ocr_info, width=width, height=height)
                seqs.append(token_seq)
        pt_seqs = stack_tensor_dict(seqs)
        return pt_seqs


class Processor(object):
    r"""Construct a GeoLayoutLM processor.

    Args:
        max_seq_length: max length for token
        max_block_num: max number of text lines (blocks or segments)
        img_processor: type of ImageProcessor.
        tokenizer: to tokenize strings.
        use_roberta_tokenizer: Whether the tokenizer is originated from RoBerta tokenizer
            (True by default).
        ocr_utils: a tool to preprocess ocr_infos.
        width: default width. It can be used only when all the images are of the same shape.
        height: default height. It can be used only when all the images are of the same shape.

    In `serialize_from_tokens`, the 3 sequences (i.e., `input_ids`, `bboxes_line`, `bboxes_word`)
        must not contain special tokens like [CLS] or [SEP].
    The boxes in `bboxes_line` and `bboxes_word` can be presented by either 2 points or 4 points.
    The value in boxes should keep original.
    Here is an example of the 3 arguments:
        ```
        input_ids ->
        [[6, 2391, 6, 31833, 6, 10132, 6, 2283, 6, 17730, 6, 2698, 152]]
        bboxes_line ->
        [[[230, 1, 353, 38], [230, 1, 353, 38], [230, 1, 353, 38], [230, 1, 353, 38],
            [230, 1, 353, 38], [230, 1, 353, 38], [230, 1, 353, 38], [230, 1, 353, 38],
            [257, 155, 338, 191], [257, 155, 338, 191], [257, 155, 338, 191], [257, 155, 338, 191],
            [257, 155, 338, 191]]]
        bboxes_word ->
        [[[231, 2, 267, 2, 267, 38, 231, 38], [231, 2, 267, 2, 267, 38, 231, 38],
            [264, 7, 298, 7, 298, 36, 264, 36], [264, 7, 298, 7, 298, 36, 264, 36],
            [293, 3, 329, 3, 329, 41, 293, 41], [293, 3, 329, 3, 329, 41, 293, 41],
            [330, 4, 354, 4, 354, 39, 330, 39], [330, 4, 354, 4, 354, 39, 330, 39],
            [258, 156, 289, 156, 289, 193, 258, 193], [258, 156, 289, 156, 289, 193, 258, 193],
            [288, 158, 321, 158, 321, 192, 288, 192], [288, 158, 321, 158, 321, 192, 288, 192],
            [321, 156, 336, 156, 336, 190, 321, 190]]]
        ```

    """

    def __init__(self,
                 max_seq_length,
                 max_block_num,
                 img_processor: ImageProcessor,
                 tokenizer=None,
                 use_roberta_tokenizer: bool = True,
                 ocr_utils: OCRUtils = None,
                 width=768,
                 height=768,
                 **kwargs):
        self.img_processor = img_processor
        self.tokenizer = tokenizer
        self.kwargs = kwargs

        self.serializer = TextLayoutSerializer(
            max_seq_length,
            max_block_num,
            tokenizer,
            width,
            height,
            use_roberta_tokenizer=use_roberta_tokenizer,
            ocr_utils=ocr_utils)

    def __call__(
        self,
        images: Union[list, np.ndarray, PIL.Image.Image, str],
        ocr_infos: List[List] = None,
        token_seqs: dict = None,
        sizes_raw: list = None,
    ):
        img_data = self.img_processor(images)
        images = img_data['images']
        ocr_infos = img_data['ocr_infos'] if ocr_infos is None else ocr_infos
        sizes_raw = img_data['sizes_raw'] if sizes_raw is None else sizes_raw
        if token_seqs is None:
            token_seqs = self.serializer(ocr_infos, sizes_raw=sizes_raw)
        else:
            token_seqs = self.serializer(
                None, sizes_raw=sizes_raw, **token_seqs)
        assert token_seqs is not None, 'token_seqs must not be NoneType!'
        batch = {}
        batch['image'] = images
        for k, v in token_seqs.items():
            batch[k] = token_seqs[k]
        return batch

    def serialize_from_tokens(self,
                              images,
                              input_ids,
                              bboxes_line,
                              bboxes_word,
                              sizes_raw=None):
        half_batch = {}
        half_batch['input_ids'] = input_ids
        half_batch['bboxes_line'] = bboxes_line
        half_batch['bboxes_word'] = bboxes_word
        return self(images, None, half_batch, sizes_raw)
