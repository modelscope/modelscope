# The implementation is adopted from Mask DINO, made publicly available under the Apache License,
# Version 2.0 at https://github.com/IDEA-Research/MaskDINO
# Part of implementation is borrowed from Mask2Former,
# https://github.com/facebookresearch/Mask2Former, under MIT license.

from typing import Any, Dict, List

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import functional as F
from torch.nn.init import constant_, kaiming_uniform_, normal_

from .ms_deform_attn import MSDeformAttn
from .position_encoding import PositionEmbeddingSine
from .utils import Conv2d, _get_activation_fn, _get_clones


# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation='relu',
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, enc_n_points)
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer,
                                                      num_encoder_layers)

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds):

        enable_mask = 0
        if masks is not None:
            for src in srcs:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [
                torch.zeros((x.size(0), x.size(2), x.size(3)),
                            device=x.device,
                            dtype=torch.bool) for x in srcs
            ]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask,
                  pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index,
                              valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        return memory, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model=256,
                 d_ffn=1024,
                 dropout=0.1,
                 activation='relu',
                 n_levels=4,
                 n_heads=8,
                 n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask=None):
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos), reference_points, src,
            spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                pos=None,
                padding_mask=None):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes,
                           level_start_index, padding_mask)

        return output


class MaskDINOEncoder(nn.Module):
    """
    This is the multi-scale encoder in detection models, also named as pixel decoder in segmentation models.
    """

    def __init__(
        self,
        input_shape: Dict[str, Any],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        common_stride: int,
        num_feature_levels: int,
        total_num_feature_levels: int,
        feature_order: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dim: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            num_feature_levels: feature scales used
            total_num_feature_levels: total feautre scales used (include the downsampled features)
            feature_order: 'low2high' or 'high2low', i.e., 'low2high' means low-resolution features
                are put in the first.
        """
        super().__init__()
        transformer_input_shape = {
            k: v
            for k, v in input_shape.items() if k in transformer_in_features
        }
        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1]['stride'])
        self.in_features = [k for k, v in input_shape
                            ]  # starting from "res2" to "res5"
        self.feature_strides = [v['stride'] for k, v in input_shape]
        self.feature_channels = [v['channels'] for k, v in input_shape]
        self.feature_order = feature_order

        if feature_order == 'low2high':
            transformer_input_shape = sorted(
                transformer_input_shape.items(), key=lambda x: -x[1]['stride'])
        else:
            transformer_input_shape = sorted(
                transformer_input_shape.items(), key=lambda x: x[1]['stride'])
        self.transformer_in_features = [k for k, v in transformer_input_shape
                                        ]  # starting from "res2" to "res5"
        transformer_in_channels = [
            v['channels'] for k, v in transformer_input_shape
        ]
        self.transformer_feature_strides = [
            v['stride'] for k, v in transformer_input_shape
        ]  # to decide extra FPN layers

        self.maskdino_num_feature_levels = num_feature_levels  # always use 3 scales
        self.total_num_feature_levels = total_num_feature_levels
        self.common_stride = common_stride

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.low_resolution_index = transformer_in_channels.index(
            max(transformer_in_channels))
        self.high_resolution_index = 0 if self.feature_order == 'low2high' else -1
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    ))
            # input projectino for downsample
            in_channels = max(transformer_in_channels)
            for _ in range(
                    self.total_num_feature_levels
                    - self.transformer_num_feature_levels):  # exclude the res2
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels,
                            conv_dim,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        nn.GroupNorm(32, conv_dim),
                    ))
                in_channels = conv_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(
                        transformer_in_channels[-1], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )
            ])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.total_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.c2_xavier_fill(self.mask_features)
        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = max(
            int(np.log2(stride) - np.log2(self.common_stride)), 1)

        lateral_convs = []
        output_convs = []

        use_bias = False
        for idx, in_channels in enumerate(
                self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = nn.GroupNorm(32, conv_dim)
            output_norm = nn.GroupNorm(32, conv_dim)

            lateral_conv = Conv2d(
                in_channels,
                conv_dim,
                kernel_size=1,
                bias=use_bias,
                norm=lateral_norm)
            output_conv = Conv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
            )
            self.c2_xavier_fill(lateral_conv)
            self.c2_xavier_fill(output_conv)
            self.add_module('adapter_{}'.format(idx + 1), lateral_conv)
            self.add_module('layer_{}'.format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def c2_xavier_fill(self, module: nn.Module) -> None:
        kaiming_uniform_(module.weight, a=1)
        if module.bias is not None:
            constant_(module.bias, 0)

    @autocast(enabled=False)
    def forward_features(self, features, masks):
        """
        Args:
            features: multi-scale features from the backbone
            masks: image mask
        Returns:
            enhanced multi-scale features and mask feature (1/4 resolution) for the decoder to produce binary mask
        """
        # backbone features
        srcs = []
        pos = []
        # additional downsampled features
        srcsl = []
        posl = []
        if self.total_num_feature_levels > self.transformer_num_feature_levels:
            smallest_feat = features[self.transformer_in_features[
                self.low_resolution_index]].float()
            _len_srcs = self.transformer_num_feature_levels
            for i in range(_len_srcs, self.total_num_feature_levels):
                if i == _len_srcs:
                    src = self.input_proj[i](smallest_feat)
                else:
                    src = self.input_proj[i](srcsl[-1])
                srcsl.append(src)
                posl.append(self.pe_layer(src))
        srcsl = srcsl[::-1]
        # Reverse feature maps
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float(
            )  # deformable detr does not support half precision
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))
        srcs.extend(
            srcsl) if self.feature_order == 'low2high' else srcsl.extend(srcs)
        pos.extend(posl) if self.feature_order == 'low2high' else posl.extend(
            pos)
        if self.feature_order != 'low2high':
            srcs = srcsl
            pos = posl
        y, spatial_shapes, level_start_index = self.transformer(
            srcs, masks, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.total_num_feature_levels
        for i in range(self.total_num_feature_levels):
            if i < self.total_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[
                    i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(
                z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0],
                                       spatial_shapes[i][1]))

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(
                out[self.high_resolution_index],
                size=cur_fpn.shape[-2:],
                mode='bilinear',
                align_corners=False)
            y = output_conv(y)
            out.append(y)
        for o in out:
            if num_cur_levels < self.total_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(out[-1]), out[0], multi_scale_features
