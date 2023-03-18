# Copyright 2022-2023 The Alibaba Fundamental Vision  Team Authors. All rights reserved.

import os

import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .blocks import (BboxRegressor, Q2VRankerStage1, Q2VRankerStage2,
                     V2QRankerStage1, V2QRankerStage2)
from .swin_transformer import SwinTransformerV2_1D


@MODELS.register_module(
    Tasks.video_temporal_grounding, module_name=Models.soonet)
class SOONet(TorchModel):
    """
        The implementation of 'Scanning Only Once: An End-to-end Framework for Fast Temporal Grounding
        in Long Videos'. The model is dynamically initialized with the following parts:
            - q2v_stage1: calculate qv_ctx_score.
            - v2q_stage1: calculate vq_ctx_score.
            - q2v_stage2: calculate qv_ctn_score.
            - v2q_stage2: calculate vq_ctn_score.
            - regressor: predict the offset of bounding box for each candidate anchor.
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """
            Initialize SOONet Model

            Args:
                model_dir: model id or path
        """
        super().__init__()
        config_path = os.path.join(model_dir, ModelFile.CONFIGURATION)
        self.config = Config.from_file(config_path).hyperparams
        nscales = self.config.nscales
        hidden_dim = self.config.hidden_dim
        snippet_length = self.config.snippet_length
        self.enable_stage2 = self.config.enable_stage2
        self.stage2_topk = self.config.stage2_topk
        self.nscales = nscales

        self.video_encoder = SwinTransformerV2_1D(
            patch_size=snippet_length,
            in_chans=hidden_dim,
            embed_dim=hidden_dim,
            depths=[2] * nscales,
            num_heads=[8] * nscales,
            window_size=[64] * nscales,
            mlp_ratio=2.,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            pretrained_window_sizes=[0] * nscales)

        self.q2v_stage1 = Q2VRankerStage1(nscales, hidden_dim)
        self.v2q_stage1 = V2QRankerStage1(nscales, hidden_dim)
        if self.enable_stage2:
            self.q2v_stage2 = Q2VRankerStage2(nscales, hidden_dim,
                                              snippet_length)
            self.v2q_stage2 = V2QRankerStage2(nscales, hidden_dim)
        self.regressor = BboxRegressor(hidden_dim, self.enable_stage2)

        # Load trained weights
        model_path = os.path.join(model_dir,
                                  'SOONet_MAD_VIT-B-32_4Scale_10C.pth')
        state_dict = torch.load(model_path, map_location='cpu')['model']
        self.load_state_dict(state_dict, strict=True)

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self, **kwargs):
        raise NotImplementedError

    def forward_test(self,
                     query_feats=None,
                     video_feats=None,
                     start_ts=None,
                     end_ts=None,
                     scale_boundaries=None,
                     **kwargs):
        """
            Obtain matching scores and bbox bias of the top-k candidate anchors, with
            pre-extracted query features and video features as input.

            Args:
                query_feats: the pre-extracted text features.
                video_feats: the pre-extracted video features.
                start_ts: the start timestamps of pre-defined multi-scale anchors.
                end_ts: the end timestamps of pre-defined multi-scale anchors.
                scale_boundaries: the begin and end anchor index for each scale in start_ts and end_ts.

            Returns:
                [final_scores, bbox_bias, starts, ends]
        """
        sent_feat = query_feats
        ctx_feats = self.video_encoder(video_feats.permute(0, 2, 1))
        qv_ctx_scores = self.q2v_stage1(ctx_feats, sent_feat)
        if self.enable_stage2:
            hit_indices = list()
            starts = list()
            ends = list()
            filtered_ctx_feats = list()
            for i in range(self.nscales):
                _, indices = torch.sort(
                    qv_ctx_scores[i], dim=1, descending=True)
                indices, _ = torch.sort(
                    torch.LongTensor(
                        list(
                            set(indices[:, :self.stage2_topk].flatten().cpu().
                                numpy().tolist()))))
                indices = indices.to(video_feats.device)
                hit_indices.append(indices)

                filtered_ctx_feats.append(
                    torch.index_select(ctx_feats[i], 1, indices))

                scale_first = scale_boundaries[i]
                scale_last = scale_boundaries[i + 1]

                filtered_start = torch.index_select(
                    start_ts[scale_first:scale_last], 0, indices)
                filtered_end = torch.index_select(
                    end_ts[scale_first:scale_last], 0, indices)
                starts.append(filtered_start)
                ends.append(filtered_end)

            starts = torch.cat(starts, dim=0)
            ends = torch.cat(ends, dim=0)

            qv_merge_scores, qv_ctn_scores, ctn_feats = self.q2v_stage2(
                video_feats, sent_feat, hit_indices, qv_ctx_scores)
            ctx_feats = filtered_ctx_feats
        else:
            ctn_feats = None
            qv_merge_scores = qv_ctx_scores
            starts = start_ts
            ends = end_ts

        bbox_bias = self.regressor(ctx_feats, ctn_feats, sent_feat)
        final_scores = torch.sigmoid(torch.cat(qv_merge_scores, dim=1))

        return final_scores, bbox_bias, starts, ends
