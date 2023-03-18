# Copyright 2022-2023 The Alibaba Fundamental Vision  Team Authors. All rights reserved.

import os
from typing import Any, Dict

import numpy as np
import torch
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.soonet import (SimpleTokenizer,
                                                  decode_video, load_clip)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_temporal_grounding,
    module_name=Pipelines.soonet_video_temporal_grounding)
class SOONetVideoTemporalGroundingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        SOONet pipeline for video temporal groundinng

        Examples:

        >>> from modelscope.pipelines import pipeline

        >>> soonet_pipeline = pipeline("video-temporal-grounding", "damo/multi-modal_soonet_video-temporal-grounding")
        >>> soonet_pipeline(
            ('a man takes food out of the refrigerator.',
             'soonet_video_temporal_grounding_test_video.mp4'))

        >>> {
        >>>    "scores": [
        >>>        0.80661213,
        >>>        0.8060084,
        >>>        0.8018835,
        >>>        0.79837507,
        >>>        0.7963626,
        >>>        0.7949013,
        >>>        0.79353744,
        >>>        0.79287416,
        >>>        0.79066336,
        >>>        0.79027915
        >>>    ],
        >>>    "tbounds": [
        >>>        [
        >>>            0,
        >>>            2.9329566955566406
        >>>        ],
        >>>        [
        >>>            1.0630402565002441,
        >>>            4.9339457750320435
        >>>        ],
        >>>        [
        >>>            300.96919429302216,
        >>>            304.8546848297119
        >>>        ],
        >>>        [
        >>>            302.96981167793274,
        >>>            306.7714672088623
        >>>        ],
        >>>        [
        >>>            0,
        >>>            5.0421366691589355
        >>>        ],
        >>>        [
        >>>            304.9119266271591,
        >>>            308.7636929154396
        >>>        ],
        >>>        [
        >>>            258.96133184432983,
        >>>            262.805901825428
        >>>        ],
        >>>        [
        >>>            122.9599289894104,
        >>>            126.86622190475464
        >>>        ],
        >>>        [
        >>>            126.94010400772095,
        >>>            130.8090701699257
        >>>        ],
        >>>        [
        >>>            121.04773849248886,
        >>>            124.79261875152588
        >>>        ]
        >>>    ]
        >>> }
        """
        super().__init__(model=model, **kwargs)

        self.model_dir = model
        self.clip = load_clip(os.path.join(self.model_dir,
                                           'ViT-B-32.pt')).to(self.device)
        self.model = self.model.float().to(self.device)
        self.model.eval()

        # Load Configuration from File
        config_path = os.path.join(self.model_dir, ModelFile.CONFIGURATION)
        self.config = Config.from_file(config_path).hyperparams
        self.nscales = self.config.nscales
        self.snippet_length = self.config.snippet_length
        self.max_anchor_length = self.snippet_length * 2**(self.nscales - 1)
        self.topk = 10
        self.fps = 5
        # Define image transform
        self.img_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                 (0.26862954, 0.26130258, 0.27577711)),
        ])
        logger.info('Init transform done')

        # Init tokenizer
        bpe_path = os.path.join(self.model_dir, 'bpe_simple_vocab_16e6.txt.gz')
        self.tokenizer = SimpleTokenizer(bpe_path)
        logger.info('Init tokenizer done')

    def pad(self, arr, pad_len):
        new_arr = np.zeros((pad_len, ), dtype=float)
        new_arr[:len(arr)] = arr
        return new_arr

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        text, video_name = input
        video_path = os.path.join(self.model_dir, video_name)
        imgs, duration = decode_video(video_path, self.fps)
        trans_imgs = list()
        for i, img in enumerate(imgs):
            trans_imgs.append(self.img_transform(img))
        imgs = trans_imgs
        token_ids = self.tokenizer.tokenize(text).to(
            self.device, non_blocking=True)
        # get the start and end timestamps of anchors
        start_ts, end_ts, scale_boundaries = list(), list(), [0]
        ori_video_length = len(imgs)
        pad_video_length = int(
            np.math.ceil(ori_video_length / self.max_anchor_length)
            * self.max_anchor_length)
        for i in range(self.config.nscales):
            anchor_length = self.config.snippet_length * (2**i)
            pad_feat_length = pad_video_length // anchor_length
            nfeats = np.math.ceil(ori_video_length / anchor_length)
            s_times = np.arange(0, nfeats).astype(np.float32) * (
                anchor_length // self.fps)
            e_times = np.arange(1, nfeats + 1).astype(np.float32) * (
                anchor_length // self.fps)
            e_times = np.minimum(duration, e_times)
            start_ts.append(self.pad(s_times, pad_feat_length))
            end_ts.append(self.pad(e_times, pad_feat_length))
            scale_boundaries.append(scale_boundaries[-1] + pad_feat_length)

        start_ts = torch.from_numpy(np.concatenate(start_ts, axis=0))
        end_ts = torch.from_numpy(np.concatenate(end_ts, axis=0))
        scale_boundaries = torch.LongTensor(scale_boundaries)
        result = {
            'token_ids': token_ids,
            'imgs': torch.stack(imgs, dim=0),
            'start_ts': start_ts,
            'end_ts': end_ts,
            'scale_boundaries': scale_boundaries
        }
        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            video_feats = self.clip.encode_image(input['imgs'].to(self.device))
            query_feats = self.clip.encode_text(input['token_ids'].to(
                self.device))
            #
            ori_video_length, feat_dim = video_feats.shape
            pad_video_length = int(
                np.math.ceil(ori_video_length / self.max_anchor_length)
                * self.max_anchor_length)
            pad_video_feats = torch.zeros((pad_video_length, feat_dim),
                                          dtype=float)
            pad_video_feats[:ori_video_length, :] = video_feats
            final_scores, bbox_bias, starts, ends = self.model(
                query_feats=query_feats.float().to(self.device),
                video_feats=pad_video_feats.unsqueeze(0).float().to(
                    self.device),
                start_ts=input['start_ts'].float().to(self.device),
                end_ts=input['end_ts'].float().to(self.device),
                scale_boundaries=input['scale_boundaries'])
        #
        final_scores = final_scores.cpu().numpy()
        bbox_bias = bbox_bias.cpu().numpy()
        starts = starts.cpu().numpy()
        ends = ends.cpu().numpy()
        pred_scores, pred_bboxes = list(), list()
        rank_id = np.argsort(final_scores[0])[::-1]
        for j in range(self.topk):
            if j >= len(rank_id):
                break
            pred_scores.append(final_scores[0, rank_id[j]])
            ori_end = float(ends[rank_id[j]])
            ori_start = float(starts[rank_id[j]])
            duration = ori_end - ori_start
            sbias = bbox_bias[0, rank_id[j], 0]
            ebias = bbox_bias[0, rank_id[j], 1]
            pred_start = max(0, ori_start + sbias * duration)
            pred_end = ori_end + ebias * duration
            pred_bboxes.append([pred_start, pred_end])

        return {
            OutputKeys.SCORES: pred_scores,
            OutputKeys.TBOUNDS: pred_bboxes
        }

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        return inputs
