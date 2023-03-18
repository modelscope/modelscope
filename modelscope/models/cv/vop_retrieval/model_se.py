# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import load_clip
from .basic_utils import get_state_dict, set_seed


@MODELS.register_module(
    Tasks.vop_retrieval, module_name=Models.vop_retrieval_model_se)
class VideoTextRetrievalModelSeries(TorchModel):
    """
        The implementation of 'VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval'.
        This model is dynamically initialized with the following parts:
            - clip: the upstream pre-trained backbone model (CLIP in this code).
                - The pretrain param (ViT-B/32) downloads from OpenAI:
                - "https://openaipublic.azureedge.net/clip/models/
                - 40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"
            - pool_frames: the frames pooling method
            - visual_prompt_learner: visual prompt
            - ImageEncoder: get image encoder
            - TextPromptLearner: text prompt
            - TextEncoder: get text encoder
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """
            Initialize a VoP Model

            Args:
                model_dir: model id or path,
        """
        super(VideoTextRetrievalModelSeries, self).__init__()
        model_path = osp.join(model_dir, 'VoPSE_msrvtt9k.pth')
        clip_arch = osp.join(model_dir, 'ViT-B-32.pt')
        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)

        self.config = Config.from_file(config_path).hyperparam
        self.clip = load_clip(name=clip_arch)

        self.pool_frames = BaselinePooling(self.config.pooling_type)

        # load param from pre-train model
        self.load_state_dict(get_state_dict(model_path))

        # eval model
        self.eval()

    def get_video_features(self, videos, return_all_frames=False):
        """
            Get video Features

            Args:
                videos: the dim is [1, 12, 3, 224, 224]
                return_all_frames: default False
        """
        batch_size = videos.shape[0]
        video_data = videos.reshape(-1, 3, self.config.input_res,
                                    self.config.input_res)

        video_features = self.clip.encode_image(video_data)

        video_features = video_features / video_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size,
                                                self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(video_features)

        if return_all_frames:
            return video_features, video_features_pooled

        return video_features_pooled

    def get_text_features(self, text_data):
        """
            Get Text Features

            Args:
                text_data: the dim is [1, 69]
        """
        text_features = self.clip.encode_text(text_data)

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)
        return text_features

    def forward(self, data, return_all_frames=False):
        """
            Dynamic Forward Function of VoP

            Args:
                data: the input data
                return_all_frames: default False
        """
        batch_size = data['video'].shape[0]
        text_data = data['text']
        video_data = data['video']
        video_data = video_data.reshape(-1, 3, self.config.input_res,
                                        self.config.input_res)

        text_features = self.clip.encode_text(text_data)
        video_features = self.clip.encode_image(video_data)

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features / video_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size,
                                                self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled


class BaselinePooling(TorchModel):
    """
        Redefined Pooling Function
    """

    def __init__(self, pooling_type):
        super(BaselinePooling, self).__init__()
        if pooling_type == 'avg':
            self.pooling_func = self._avg_pooling
        else:
            raise NotImplementedError

    def _avg_pooling(self, video_embeds):
        """
            Pooling mean of frames

            Args:
                video_embeds: the input video embedding with [1, 12, 512].

            Returns:
                video_embeds_pooled: num_vids x embed_dim
        """
        video_embeds_pooled = video_embeds.mean(dim=1)
        return video_embeds_pooled

    def forward(self, video_embeds):
        return self.pooling_func(video_embeds)
