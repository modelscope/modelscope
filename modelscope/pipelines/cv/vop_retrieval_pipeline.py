# Copyright (c) Alibaba, Inc. and its affiliates.

import gzip
import math
import os
import os.path as osp
import pickle
import random
from collections import defaultdict, deque
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.cv.vop_retrieval import (LengthAdaptiveTokenizer, VoP,
                                                init_transform_dict, load_data,
                                                load_frames_from_video)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import load_image
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.vop_retrieval, module_name=Pipelines.vop_retrieval)
class VopRetrievalPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a vop pipeline for retrieval
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)

        # [from pretrain] load model
        self.model = Model.from_pretrained('damo/cv_vit-b32_retrieval_vop').to(
            self.device)
        logger.info('load model done')

        # others: load transform
        self.local_pth = model
        self.cfg = Config.from_file(osp.join(model, ModelFile.CONFIGURATION))
        self.img_transform = init_transform_dict(
            self.cfg.hyperparam.input_res)['clip_test']
        logger.info('load transform done')

        # others: load tokenizer
        bpe_path = gzip.open(osp.join(
            model,
            'bpe_simple_vocab_16e6.txt.gz')).read().decode('utf-8').split('\n')
        self.tokenizer = LengthAdaptiveTokenizer(self.cfg.hyperparam, bpe_path)
        logger.info('load tokenizer done')

        # others: load dataset
        self.database = load_data(
            osp.join(model, 'VoP_msrvtt9k_features.pkl'), self.device)
        logger.info('load database done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            if '.mp4' in input:
                query = []
                for video_path in [input]:
                    video_path = osp.join(self.local_pth, video_path)
                    imgs, idxs = load_frames_from_video(
                        video_path, self.cfg.hyperparam.num_frames,
                        self.cfg.hyperparam.video_sample_type)
                    imgs = self.img_transform(imgs)
                    query.append(imgs)
                query = torch.stack(
                    query, dim=0).to(
                        self.device, non_blocking=True)
                mode = 'v2t'
            else:
                query = self.tokenizer(
                    input, return_tensors='pt', padding=True, truncation=True)
                if isinstance(query, torch.Tensor):
                    query = query.to(self.device, non_blocking=True)
                else:
                    query = {
                        key: val.to(self.device, non_blocking=True)
                        for key, val in query.items()
                    }
                mode = 't2v'
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        result = {'input_data': query, 'mode': mode}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        text_embeds, vid_embeds_pooled, vid_ids, texts = self.database
        with torch.no_grad():
            if input['mode'] == 't2v':
                query_feats = self.model.get_text_features(input['input_data'])
                score = query_feats @ vid_embeds_pooled.T
                retrieval_idxs = torch.topk(
                    score, k=self.cfg.hyperparam.topk,
                    dim=-1)[1].cpu().numpy()
                res = np.array(vid_ids)[retrieval_idxs]
            elif input['mode'] == 'v2t':
                query_feats = self.model.get_video_features(
                    input['input_data'])
                score = query_feats @ text_embeds.T
                retrieval_idxs = torch.topk(
                    score, k=self.cfg.hyperparam.topk,
                    dim=-1)[1].cpu().numpy()
                res = np.array(texts)[retrieval_idxs]
            results = {'output_data': res, 'mode': input['mode']}
            return results

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
