# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import time
import traceback
from typing import Any, Dict, Optional

import json
import numpy as np
import torch
from PIL import Image
from transformers import BertTokenizer

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.metainfo import Pipelines
from modelscope.pipelines import Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import load_image
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, Frameworks,
                                       Invoke, Tasks)
from modelscope.utils.logger import get_logger

logger = get_logger()


def cost(end, begin):
    return '{:.2f}ms'.format((end - begin) * 1000)


class Config:
    SCALE = 1 / 255.0
    MEAN = np.require([0.485, 0.456, 0.406], dtype=np.float32)[:, np.newaxis,
                                                               np.newaxis]
    STD = np.require([0.229, 0.224, 0.225], dtype=np.float32)[:, np.newaxis,
                                                              np.newaxis]

    # RESIZE_HEIGHT = int(224*1.14)
    RESIZE_HEIGHT = int(256)
    # RESIZE_WIDTH = int(224*1.14)
    RESIZE_WIDTH = int(256)
    CROP_SIZE = 224


def pre_processor(img):
    img = img.convert('RGB')

    w, h = img.size
    if (w <= h and w == Config.RESIZE_WIDTH) \
            or (h <= w and h == Config.RESIZE_WIDTH):
        img = img
    if w < h:
        ow = Config.RESIZE_WIDTH
        oh = int(Config.RESIZE_WIDTH * h / w)
        img = img.resize((ow, oh), Image.BILINEAR)
    else:
        oh = Config.RESIZE_WIDTH
        ow = int(Config.RESIZE_WIDTH * w / h)
        img = img.resize((ow, oh), Image.BILINEAR)
    w, h = img.size
    crop_top = int(round((h - Config.CROP_SIZE) / 2.))
    crop_left = int(round((w - Config.CROP_SIZE) / 2.))
    img = img.crop((crop_left, crop_top, crop_left + Config.CROP_SIZE,
                    crop_top + Config.CROP_SIZE))
    _img = np.array(img, dtype=np.float32)
    _img = np.require(_img.transpose((2, 0, 1)), dtype=np.float32)
    _img *= Config.SCALE
    _img -= Config.MEAN
    _img /= Config.STD
    return _img


class GridVlpPipeline(Pipeline):
    """ Pipeline for gridvlp, including classification and embedding."""

    def __init__(self, model_name_or_path: str, **kwargs):
        """ Pipeline for gridvlp, including classification and embedding.
        Args:
            model: path to local model directory.
        """
        # download model from modelscope to local model dir
        logger.info(f'load checkpoint from modelscope {model_name_or_path}')
        if osp.exists(model_name_or_path):
            local_model_dir = model_name_or_path
        else:
            invoked_by = '%s/%s' % (Invoke.KEY, Invoke.PIPELINE)
            local_model_dir = snapshot_download(
                model_name_or_path,
                DEFAULT_MODEL_REVISION,
                user_agent=invoked_by)
        self.local_model_dir = local_model_dir

        # load model from cpu and torch jit model
        logger.info(f'load model from {local_model_dir}')
        self.model = torch.jit.load(
            osp.join(local_model_dir, 'pytorch_model.pt'))
        self.framework = Frameworks.torch
        self.device_name = 'cpu'
        self._model_prepare = True
        self._auto_collate = False

        # load tokenizer
        logger.info(f'load tokenizer from {local_model_dir}')
        self.tokenizer = BertTokenizer.from_pretrained(local_model_dir)

    def preprocess(self, inputs: Dict[str, Any], max_seq_length=49):
        # fetch input params
        image = inputs.get('image', '')
        text = inputs.get('text', '')

        s1 = time.time()

        # download image and preprocess
        try:
            # load PIL image
            img = load_image(image)
            s2 = time.time()

            # image preprocess
            image_data = pre_processor(img)
            s3 = time.time()

        except Exception:
            image_data = np.zeros((3, 224, 224), dtype=np.float32)
            s2 = time.time()
            s3 = time.time()
            logger.info(traceback.print_exc())

        # text process
        if text is None or text.isspace() or not text.strip():
            logger.info('text is empty!')
            text = ''
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_seq_length)

        s4 = time.time()

        logger.info(f'example. text: {text} image: {image}')
        logger.info(
            f'preprocess. Img_Download:{cost(s2, s1)}, Img_Pre:{cost(s3, s2)}, Txt_Pre:{cost(s4, s3)}'
        )

        input_dict = {
            'image': image_data,
            'input_ids': inputs['input_ids'],
            'input_mask': inputs['attention_mask'],
            'segment_ids': inputs['token_type_ids']
        }
        return input_dict


@PIPELINES.register_module(
    Tasks.visual_question_answering,
    module_name=Pipelines.gridvlp_multi_modal_classification)
class GridVlpClassificationPipeline(GridVlpPipeline):
    """ Pipeline for gridvlp classification, including cate classfication and
    brand classification.

    Example:

    ```python
    >>> from modelscope.pipelines.multi_modal.gridvlp_pipeline import \
    GridVlpClassificationPipeline

    >>> pipeline = GridVlpClassificationPipeline('rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-cate')
    >>> output = pipeline({'text': '女装快干弹力轻型短裤448575',\
        'image':'https://yejiabo-public.oss-cn-zhangjiakou.aliyuncs.com/alinlp/clothes.png'})
    >>> output['text'][0]
    {'label': {'cate_name': '休闲裤', 'cate_path': '女装>>裤子>>休闲裤>>休闲裤'}, 'score': 0.4146, 'rank': 0}

    ```
    """

    def __init__(self, model_name_or_path: str, **kwargs):
        """ Pipeline for gridvlp classification, including cate classfication and
    brand classification.
        Args:
            model: path to local model directory.
        """
        super().__init__(model_name_or_path, **kwargs)

        # load label mapping
        logger.info(f'load label mapping from {self.local_model_dir}')
        self.label_mapping = json.load(
            open(osp.join(self.local_model_dir, 'label_mapping.json')))

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        s4 = time.time()

        box_tensor = torch.zeros(1, dtype=torch.float32)

        output = self.model(
            torch.tensor(inputs['image']).unsqueeze(0),
            box_tensor.unsqueeze(0),
            torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0),
            torch.tensor(inputs['input_mask'], dtype=torch.long).unsqueeze(0),
            torch.tensor(inputs['segment_ids'], dtype=torch.long).unsqueeze(0))
        output = output[0].detach().numpy()

        s5 = time.time()

        logger.info(f'forward. Infer:{cost(s5, s4)}')

        # 返回结果
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        s5 = time.time()
        output = inputs
        index = np.argsort(-output)
        out_sort = output[index]

        top_k = []
        for i in range(min(10, len(self.label_mapping))):
            label = self.label_mapping[str(index[i])]
            top_k.append({
                'label': label,
                'score': round(float(out_sort[i]), 4),
                'rank': i
            })

        s6 = time.time()
        logger.info(f'postprocess. Post: {cost(s6, s5)}')
        return {'text': top_k}


@PIPELINES.register_module(
    Tasks.multi_modal_embedding,
    module_name=Pipelines.gridvlp_multi_modal_embedding)
class GridVlpEmbeddingPipeline(GridVlpPipeline):
    """ Pipeline for gridvlp embedding. These only generate unified multi-modal
    embeddings and output it in `text_embedding` or `img_embedding`.

    Example:

    ```python
    >>> from modelscope.pipelines.multi_modal.gridvlp_pipeline import \
    GridVlpEmbeddingPipeline

    >>> pipeline = GridVlpEmbeddingPipeline('rgtjf1/multi-modal_gridvlp_classification_chinese-base-ecom-embedding')
    >>> outputs = pipeline({'text': '女装快干弹力轻型短裤448575',\
        'image':'https://yejiabo-public.oss-cn-zhangjiakou.aliyuncs.com/alinlp/clothes.png'})
    >>> outputs["text_embedding"].shape
    (768,)

    ```
    """

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        s4 = time.time()

        box_tensor = torch.zeros(1, dtype=torch.float32)

        output = self.model(
            torch.tensor(inputs['image']).unsqueeze(0),
            box_tensor.unsqueeze(0),
            torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0),
            torch.tensor(inputs['input_mask'], dtype=torch.long).unsqueeze(0),
            torch.tensor(inputs['segment_ids'], dtype=torch.long).unsqueeze(0))
        s5 = time.time()

        output = output[0].detach().numpy()

        s6 = time.time()
        logger.info(f'forward. Infer:{cost(s5, s4)}, Post: {cost(s6, s5)}')
        # 返回结果
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        outputs = {
            'img_embedding': inputs,
            'text_embedding': inputs,
        }
        return outputs
