# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import logging
import math
import os
import re
import sys

import json
import torch
import torch.distributed as dist
import torch.nn as nn
from torchvision.ops import roi_align

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.multi_modal.vldoc.conv_fpn_trans import FPNTrans
from modelscope.models.multi_modal.vldoc.modeling_layout_roberta import (
    LayoutRobertaModel, LayoutRobertaPreTrainedModel)
from modelscope.models.multi_modal.vldoc.transformer_local import (
    TransformerDecoder, TransformerDecoderLayer)
from modelscope.utils.constant import ModeKeys, ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['VLDocForDocVLEmbedding']


class GeoVLDocModelOutputs(object):

    def __init__(
        self,
        text_features,
        text_mm_features,
        block_vis_features,
        block_vis_mm_features,
        image_mm_features,
    ):
        # [batch size, sequence length, hidden size]
        self.text_features = text_features
        # [batch size, sequence length, hidden size]
        self.text_mm_features = text_mm_features
        # [batch size, block num, hidden size]
        self.block_vis_features = block_vis_features
        # [batch size, block num, hidden size]
        self.block_vis_mm_features = block_vis_mm_features
        # [batch size, hidden size]
        self.image_mm_features = image_mm_features


class GeoVLDocModel(LayoutRobertaPreTrainedModel):

    def __init__(self, config, hard_negtive_sampling=False):
        super().__init__(config)
        self.config = config
        self.hard_negtive_sampling = hard_negtive_sampling

        if getattr(self.config, 'architectures', None):
            if self.config.architectures[0] == 'LayoutRobertaModel':
                self.text_encoder = LayoutRobertaModel(config)
            else:
                self.text_encoder = LayoutRobertaModel(config)
        else:
            self.text_encoder = LayoutRobertaModel(config)
        self.visual_encoder = FPNTrans(
            img_size=self.config.image_size, inner_vit=False)
        self.pool = nn.AdaptiveAvgPool2d([1, 1])
        self.vis_linear = nn.Linear(256, self.config.hidden_size)

        cross_modal_text_layer = TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
            self_attn=True)
        self.cross_modal_text = TransformerDecoder(cross_modal_text_layer, 1)

        cross_modal_visual_layer = TransformerDecoderLayer(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.intermediate_size,
            self_attn=True)
        self.cross_modal_visual = TransformerDecoder(cross_modal_visual_layer,
                                                     1)

        self.init_weights()

    def from_pretrained(self, ckpt_path: str):
        state_dict = torch.load(ckpt_path, map_location='cpu')
        state_dict_new = {}
        for k, v in state_dict.items():
            k = k.replace('geo_vl_doc_model.', '')
            state_dict_new[k] = v
        self.load_state_dict(state_dict_new)

    def forward(self,
                input_ids=None,
                image=None,
                bbox=None,
                bbox_4p_normalized=None,
                attention_mask=None,
                first_token_idxes=None,
                first_token_idxes_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):

        batch_size, seq_len = input_ids.shape

        return_dict = (
            return_dict
            if return_dict is not None else self.config.use_return_dict)

        kwargs['line_bbox'] = bbox
        # ################ get text representation ################
        if self.config.architectures[0] == 'LayoutRobertaModel':
            outputs = self.text_encoder(
                input_ids,
                bbox=bbox_4p_normalized,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs)
        else:
            outputs = self.text_encoder(
                input_ids,
                bbox=bbox_4p_normalized,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs)

        # sequence_output: [batch_size, seq_len, hidden_size]
        # pooled_output: [batch_size, hidden_size]
        sequence_output, pooled_output = outputs[:2]

        # ################ get visual representation ################
        _, num_first = first_token_idxes.shape
        B_batch_dim = torch.arange(
            0, batch_size,
            device=input_ids.device).reshape(batch_size,
                                             1).expand(batch_size, num_first)

        feature_bbox = bbox[B_batch_dim, first_token_idxes]
        _, block_num, _ = feature_bbox.shape

        visual_out = self.visual_encoder(image)
        batch_idxs = torch.arange(
            0, batch_size, device=sequence_output.device).reshape(
                batch_size, 1).expand(batch_size, block_num).unsqueeze(-1)

        # [batch_size*block_num, 5]
        batch_idx_with_bbox = torch.cat(
            (batch_idxs, feature_bbox),
            2).reshape(batch_size * block_num,
                       5).to(dtype=visual_out['feat_ms'].dtype)

        if visual_out['feat_ms'].dtype == torch.float16:
            # [batch_size*block_num, 256, 1, 1]
            blk_vis_features = roi_align(
                visual_out['feat_ms'].to(torch.float32),
                batch_idx_with_bbox.to(torch.float32),
                1,
                spatial_scale=visual_out['feat_ms'].size(-1) / 1000.0)
            blk_vis_features = blk_vis_features.to(
                dtype=visual_out['feat_ms'].dtype)
        else:
            blk_vis_features = roi_align(
                visual_out['feat_ms'],
                batch_idx_with_bbox.to(torch.float32),
                1,
                spatial_scale=visual_out['feat_ms'].size(-1) / 1000.0)

        # [batch_size*block_num, 256]
        blk_vis_features = blk_vis_features.squeeze(2).squeeze(2).reshape(
            batch_size, block_num, 256)

        # visual block features:
        # blk_vis_features: [batch_size, block_num, hidden_size]
        blk_vis_features = self.vis_linear(blk_vis_features)
        blk_vis_features = blk_vis_features * first_token_idxes_mask.unsqueeze(
            2)
        # [batch_size, 256]
        full_img_features = self.pool(
            visual_out['feat_ms']).squeeze(2).squeeze(2)
        # [batch_size, hidden_size]
        full_img_features = self.vis_linear(full_img_features).unsqueeze(1)

        # ################ multi-modal fusion ################

        # cross attention inputs
        vis_inps = torch.cat((full_img_features, blk_vis_features), 1)

        glb_feat_attn = torch.ones((batch_size, 1)).to(input_ids.device)

        vis_mask = torch.cat((glb_feat_attn, first_token_idxes_mask), 1)

        # When we use transformer in torch.nn, the input size is
        # [seq_len, batch_size, hidden_size]
        # In attention_mask, 1 denotes masked
        new_attention_mask = (1 - attention_mask) > 0
        new_vis_mask = (1 - vis_mask) > 0

        text_mm_feat = self.cross_modal_text(
            tgt=sequence_output.transpose(0, 1),
            memory=vis_inps.transpose(0, 1),
            tgt_key_padding_mask=new_attention_mask,
            memory_key_padding_mask=new_vis_mask)

        vis_mm_feat = self.cross_modal_visual(
            tgt=vis_inps.transpose(0, 1),
            memory=sequence_output.transpose(0, 1),
            tgt_key_padding_mask=new_vis_mask,
            memory_key_padding_mask=new_attention_mask,
        )

        # [batch_size, seq_len, hidden_size]
        text_mm_feat = text_mm_feat.transpose(0, 1)
        # [batch_size, 1+block_num, hidden_size]
        vis_mm_feat = vis_mm_feat.transpose(0, 1)

        # image_mm_features = vis_mm_feat[:, 0, :]
        block_vis_mm_features = vis_mm_feat[:, 1:]

        return GeoVLDocModelOutputs(
            text_features=sequence_output,
            text_mm_features=text_mm_feat,
            block_vis_features=blk_vis_features,
            block_vis_mm_features=block_vis_mm_features,
            image_mm_features=vis_mm_feat,
        )


@MODELS.register_module(Tasks.document_vl_embedding, module_name=Models.vldoc)
class VLDocForDocVLEmbedding(TorchModel):
    """
    Generate multi-modal document embeddings in segment-level and token-level.

    Args:
        model_dir:
            the path in model hub, e.g., 'damo/multi-modal_convnext-roberta-base_vldoc-embedding'
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        super().__init__(model_dir=model_dir, *args, **kwargs)

        # Initialize the model.
        from modelscope.models.multi_modal.vldoc.modeling_layout_roberta import LayoutRobertaConfig
        model_cfg_path = os.path.join(model_dir, 'config.json')
        logger.info('Loading config file from {}'.format(model_cfg_path))
        assert os.path.exists(model_cfg_path)
        self.config = LayoutRobertaConfig.from_json_file(model_cfg_path)
        self.doc_model = GeoVLDocModel(self.config)

        # restore the pretrained weight
        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        assert os.path.exists(model_path)
        self.doc_model.from_pretrained(model_path)
        logger.info('Loading model from {}'.format(model_path))

        # Initialize the tokenizer.
        from modelscope.models.multi_modal.vldoc.tokenization import VLDocXLMTokenizer
        tokenizer_path = os.path.join(model_dir, ModelFile.TOKENIZER_FOLDER)
        self.tokenizer = VLDocXLMTokenizer.from_pretrained(tokenizer_path)

        # place the model
        self.device = 'cuda:{}'.format(int(os.environ.get(
            'LOCAL_RANK', 0))) if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            self.doc_model.to(self.device)
            logger.info('Use GPU {} for finetuning & inference'.format(
                int(os.environ.get('LOCAL_RANK', 0))))
        else:
            self.doc_model.float()
            logger.info('Use CPU for finetuning & inference')

    def forward(self,
                input_ids=None,
                image=None,
                bbox=None,
                bbox_4p_normalized=None,
                attention_mask=None,
                first_token_idxes=None,
                first_token_idxes_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        """
        Args:
            - input_ids: :math:`(B, T, E)`, the input tokens, where B is the batch size,
              T is the max token size, E is the embedding dimension.
            - image: :math:`(B, C, H, W)`, normalized images.
            - bbox: :math:`(B, T, 4)`, segment boxes denoted by top-left and bottom-right
              vertexes whose values are normalized to [0, 1000).
            - bbox_4p_normalized: :math:`(B, T, 8)`, word boxes denoted by 4 vertexes, whose
              values are normalized to [0, 1).
            - attention_mask: :math:`(B, T)`, mask for input tokens, where 0 means masked.
            - first_token_idxes: :math:`(B, S)`, indexes of the corresponding first tokens
              of all segments, where S is the max segment size.
            - first_token_idxes_mask: :math:`(B, S)`, mask for segments, where 0 means masked.
        Optional:
            - line_rank_id: :math:`(B, T)`, orders of segments.
            - line_rank_inner_id: :math:`(B, T)`, BIE-like tags.

        To be more specific, please refer to the class `TextLayoutSerializer` in
          `modelscope/models/multi_modal/vldoc/processing.py`.
        """

        vldoc_outputs = self.doc_model(
            input_ids=input_ids,
            image=image,
            bbox=bbox,
            bbox_4p_normalized=bbox_4p_normalized,
            attention_mask=attention_mask,
            first_token_idxes=first_token_idxes,
            first_token_idxes_mask=first_token_idxes_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs)

        return dict(
            img_embedding=vldoc_outputs.image_mm_features,
            text_embedding=vldoc_outputs.text_mm_features,
        )


def init_pretrained_weight(
    model,
    pretrained_model_path,
    state_dict=None,
    cache_dir=None,
    init_backbone='roberta',
):
    if state_dict is None:
        state_dict = torch.load(pretrained_model_path, map_location='cpu')

    old_keys = []
    new_keys = []
    state_dict_keys = list(state_dict.keys())

    if init_backbone == 'roberta':
        for i in range(len(state_dict_keys)):
            key = state_dict_keys[i]
            new_key = None

            if key.startswith('roberta.'):
                new_key = key.replace('roberta.',
                                      'geo_vl_doc_model.text_encoder.')
                key = copy.deepcopy(new_key)

            if new_key:
                old_keys.append(state_dict_keys[i])
                new_keys.append(new_key)

    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    start_prefix = ''
    if not hasattr(model, 'geo_vl_doc_model') and any(
            s.startswith('geo_vl_doc_model.') for s in state_dict.keys()):
        start_prefix = 'geo_vl_doc_model.'
    load(model, prefix=start_prefix)
    if len(missing_keys) > 0:
        logger.info(
            'Weights of {} not initialized from pretrained model: {}'.format(
                model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        logger.info('Weights from pretrained model not used in {}: {}'.format(
            model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        raise RuntimeError(
            'Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, '\n\t'.join(error_msgs)))

    return model
