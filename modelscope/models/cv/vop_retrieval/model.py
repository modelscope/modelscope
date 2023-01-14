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
    Tasks.vop_retrieval, module_name=Models.vop_retrieval_model)
class VoP(TorchModel):
    """
        The implementation of 'VoP: Text-Video Co-operative Prompt Tuning for Cross-Modal Retrieval'.
        This model is dynamically initialized with the following parts:
            - clip: the upstream pre-trained backbone model (CLIP in this code)
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
        super(VoP, self).__init__()
        model_path = osp.join(model_dir, 'VoP_msrvtt9k.pth')
        clip_arch = osp.join(model_dir, 'ViT-B-32.pt')
        config_path = osp.join(model_dir, ModelFile.CONFIGURATION)

        self.config = Config.from_file(config_path).hyperparam
        self.clip = load_clip(name=clip_arch)

        self.config.vpt_layers = list(
            range(self.clip.visual.transformer.layers))
        self.config.tpt_layers = list(range(self.clip.transformer.layers))

        self.pool_frames = BaselinePooling(self.config.pooling_type,
                                           self.config)

        self.visual_prompt_learner = VisualPromptLearner(
            self.clip, self.config)
        self.image_encoder = ImageEncoder(self.clip, self.config)

        self.text_prompt_learner = TextPromptLearner(self.clip, self.config)
        self.text_encoder = TextEncoder(self.clip, self.config)

        # load param from pre-train model
        self.load_state_dict(get_state_dict(model_path))
        self.eval()

        # set seed
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        set_seed(self.config.seed)

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

        visual_prompts = self.visual_prompt_learner()
        video_features = self.image_encoder(visual_prompts, video_data)

        video_features = video_features / video_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size,
                                                self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(None, video_features)

        if return_all_frames:
            return video_features, video_features_pooled

        return video_features_pooled

    def get_text_features(self, text_data):
        """
            Get Text Features

            Args:
                text_data: the dim is [1, 69]
        """
        text_prompts = self.text_prompt_learner()
        text_features = self.text_encoder(text_prompts, text_data)

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

        visual_prompts = self.visual_prompt_learner()
        video_features = self.image_encoder(visual_prompts, video_data)

        text_prompts = self.text_prompt_learner()
        text_features = self.text_encoder(text_prompts, text_data)

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features / video_features.norm(
            dim=-1, keepdim=True)
        video_features = video_features.reshape(batch_size,
                                                self.config.num_frames, -1)

        video_features_pooled = self.pool_frames(text_features, video_features)

        if return_all_frames:
            return text_features, video_features, video_features_pooled

        return text_features, video_features_pooled


class BaselinePooling(TorchModel):
    """
        Redefined Pooling Function
    """

    def __init__(self, pooling_type, config):
        super(BaselinePooling, self).__init__()
        if pooling_type == 'avg':
            self.pooling_func = self._avg_pooling
        else:
            raise NotImplementedError

    def _avg_pooling(self, text_embeds, video_embeds):
        """
            Pooling mean of frames

            Args:
                text_embeds: the input text embedding which is None here.
                video_embeds: the input video embedding with [1, 12, 512].

            Returns:
                video_embeds_pooled: num_vids x embed_dim
        """
        video_embeds_pooled = video_embeds.mean(dim=1)
        return video_embeds_pooled

    def forward(self, text_embeds, video_embeds):
        return self.pooling_func(text_embeds, video_embeds)


class VisualPromptLearner(TorchModel):
    """
        The implementation of visual prompt.
        This module is used to define the learnable prompt parameters:
            the number of tokens is 8,
            the prompt dimension is 768,
            and the initialization weight std used is 0.02.
    """

    def __init__(self, clip_model, config):
        super(VisualPromptLearner, self).__init__()

        vp_token_num = config.vp_token_num
        vp_dim = clip_model.visual.ln_post.weight.shape[0]
        dtype = clip_model.dtype

        visual_prompts = torch.empty(
            len(config.vpt_layers), 1, vp_token_num, vp_dim, dtype=dtype)
        nn.init.normal_(visual_prompts, std=0.02)
        self.visual_prompts = nn.Parameter(visual_prompts)

    def forward(self):
        vp = self.visual_prompts
        return vp


class TextPromptLearner(TorchModel):
    """
        The implementation of visual prompt.
        This module is used to define the learnable prompt parameters:
            the number of tokens is 4,
            the prompt dimension is 512,
            and the initialization weight std used is 0.02.
    """

    def __init__(self, clip_model, config):
        super(TextPromptLearner, self).__init__()

        tp_prefix_token_num = config.tp_prefix_token_num
        tp_suffix_token_num = config.tp_suffix_token_num
        assert tp_prefix_token_num >= 0 and tp_suffix_token_num >= 0
        tp_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype

        text_prompts = torch.empty(
            len(config.tpt_layers),
            tp_prefix_token_num + tp_suffix_token_num,
            tp_dim,
            dtype=dtype)
        nn.init.normal_(text_prompts, std=0.02)

        self.text_prompts = nn.Parameter(text_prompts)
        self.tp_prefix_token_num = tp_prefix_token_num
        self.tp_suffix_token_num = tp_suffix_token_num

    def forward(self):
        return (self.text_prompts[:, :self.tp_prefix_token_num, :],
                self.text_prompts[:, self.tp_prefix_token_num:, :])


class ImageEncoder(TorchModel):
    """
        The implementation of image encoder.
        This module is used to obtain the features of each frame of the video.
    """

    def __init__(self, clip_model, config):
        super(ImageEncoder, self).__init__()

        self.config = config
        self.vpt_layers = config.vpt_layers
        self.vp_token_num = config.vp_token_num
        self.num_frames = config.num_frames

        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.ln_pre = clip_model.visual.ln_pre

        self.transformer = clip_model.visual.transformer

        self.ln_post = clip_model.visual.ln_post
        self.proj = clip_model.visual.proj

    def forward(self, visual_prompts, x):
        """
            The forward function of image encoder.

            Args:
                visual_prompts: the visual prompt, dim is [12, 1, 8, 768]
                x: the input data, dim is [12, 3, 224, 224]

            Returns:
                x: the output data, dim is [12, 512]
        """
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = x.reshape(batch_size, x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x_1 = self.class_embedding.to(x.dtype)
        x_2 = torch.zeros(
            batch_size, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x_1 = x_1 + x_2
        x = torch.cat([x_1, x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        for i_layer in range(self.transformer.layers):
            if i_layer in self.vpt_layers:
                i_prompt = self.vpt_layers.index(i_layer)
                cur_layer_vp = visual_prompts[i_prompt, :, :, :].repeat(
                    batch_size, 1, 1)
                x = torch.cat([x[:, :1, :], cur_layer_vp, x[:, 1:, :]], dim=1)

            if i_layer == 0:
                x = self.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.transformer.resblocks[i_layer](x)
            x = x.permute(1, 0, 2)

            if i_layer + 1 in self.vpt_layers:
                x = torch.cat([x[:, :1, :], x[:, 1 + self.vp_token_num:, :]],
                              dim=1)

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class TextEncoder(TorchModel):
    """
        The implementation of text encoder.
        This module is used to obtain the features of each word of the sentence.
    """

    def __init__(self, clip_model, config):
        super(TextEncoder, self).__init__()
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

        self.tpt_layers = config.tpt_layers
        assert 0 in self.tpt_layers
        self.tp_prefix_token_num = config.tp_prefix_token_num
        self.tp_suffix_token_num = config.tp_suffix_token_num
        self.tp_token_num = config.tp_prefix_token_num + config.tp_suffix_token_num

    def forward(self, text_prompts, text):
        """
            The forward function of text encoder.

            Args:
                text_prompts: the text prompt, dim is 2 x [12, 4, 512]
                text: the input data, dim is [1, 69]

            Returns:
                x: the output data, dim is [1, 512]
        """
        x = self.token_embedding(text).type(self.dtype)
        batch_size = x.shape[0]
        prompt_prefix, prompt_suffix = text_prompts

        for i_layer in range(self.transformer.layers):
            if i_layer in self.tpt_layers:
                i_prompt = self.tpt_layers.index(i_layer)
                if self.tp_prefix_token_num > 0:
                    cur_layer_tp_prefix = prompt_prefix[i_prompt:i_prompt
                                                        + 1, :, :].expand(
                                                            batch_size, -1, -1)
                    x = torch.cat(
                        [x[:, :1, :], cur_layer_tp_prefix, x[:, 1:, :]], dim=1)
                if self.tp_suffix_token_num > 0:
                    cur_layer_tp_suffix = prompt_suffix[i_prompt:i_prompt
                                                        + 1, :, :].expand(
                                                            batch_size, -1, -1)
                    x = torch.cat(
                        [x[:, :-1, :], cur_layer_tp_suffix, x[:, -1:, :]],
                        dim=1)

            if i_layer == 0:
                x = x + self.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.transformer.resblocks[i_layer](x)
            x = x.permute(1, 0, 2)

            if i_layer + 1 in self.tpt_layers:
                temp_1 = x[:, :1, :]
                temp_2 = x[:, 1 + self.tp_prefix_token_num:-1
                           - self.tp_suffix_token_num, :]
                temp_3 = x[:, -1:, :]
                temp = torch.cat([temp_1, temp_2, temp_3], dim=1)
                x = temp

        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]),
              text.argmax(dim=-1) + self.tp_token_num] @ self.text_projection

        return x
