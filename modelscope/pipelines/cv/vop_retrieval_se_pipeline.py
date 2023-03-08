# Copyright (c) Alibaba, Inc. and its affiliates.

import gzip
import os.path as osp
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.cv.vop_retrieval import (LengthAdaptiveTokenizer,
                                                init_transform_dict, load_data,
                                                load_frames_from_video)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.vop_retrieval, module_name=Pipelines.vop_retrieval_se)
class VopRetrievalSEPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        r""" Card VopRetrievalSE Pipeline.

        Examples:
        >>>
        >>>   from modelscope.pipelines import pipeline
        >>>   vop_pipeline = pipeline(Tasks.vop_retrieval,
        >>>            model='damo/cv_vit-b32_retrieval_vop_bias')
        >>>
        >>>   # IF DO TEXT-TO-VIDEO:
        >>>   input_text = 'a squid is talking'
        >>>   result = vop_pipeline(input_text)
        >>>   result:
        >>>   {'output_data': array([['video8916']], dtype='<U9'),'mode': 't2v'}
        >>>
        >>>   # IF DO VIDEO-TO-TEXT:
        >>>   input_video = 'video10.mp4'
        >>>   result = vop_pipeline(input_video)
        >>>   result:
        >>>   {'output_data': array([['assorted people are shown holding cute pets']], dtype='<U163'), 'mode': 'v2t'}
        >>>
        """
        super().__init__(model=model, **kwargs)

        # [from pretrain] load model
        self.model = Model.from_pretrained(model).to(self.device)
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
        if 'vop_bias' in model:
            self.database = load_data(
                osp.join(model, 'Bias_msrvtt9k_features.pkl'), self.device)
        elif 'vop_partial' in model:
            self.database = load_data(
                osp.join(model, 'Partial_msrvtt9k_features.pkl'), self.device)
        elif 'vop_proj' in model:
            self.database = load_data(
                osp.join(model, 'Proj_msrvtt9k_features.pkl'), self.device)
        else:
            self.database = load_data(
                osp.join(model, 'VoP_msrvtt9k_features.pkl'), self.device)
        logger.info('load database done')

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
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

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
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

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        return inputs
