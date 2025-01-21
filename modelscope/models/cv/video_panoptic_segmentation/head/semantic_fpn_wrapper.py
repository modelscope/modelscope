# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, normal_init
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.builder import NECKS


@NECKS.register_module()
class SemanticFPNWrapper(nn.Module):
    """
    Implementation of Semantic FPN used in Panoptic FPN.

    Args:
        in_channels ([type]): [description]
        feat_channels ([type]): [description]
        out_channels ([type]): [description]
        start_level ([type]): [description]
        end_level ([type]): [description]
        cat_coors (bool, optional): [description]. Defaults to False.
        fuse_by_cat (bool, optional): [description]. Defaults to False.
        conv_cfg ([type], optional): [description]. Defaults to None.
        norm_cfg ([type], optional): [description]. Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 start_level,
                 end_level,
                 cat_coors=False,
                 positional_encoding=None,
                 cat_coors_level=3,
                 fuse_by_cat=False,
                 return_list=False,
                 upsample_times=3,
                 with_pred=True,
                 num_aux_convs=0,
                 act_cfg=dict(type='ReLU', inplace=True),
                 out_act_cfg=dict(type='ReLU'),
                 conv_cfg=None,
                 norm_cfg=None):
        super(SemanticFPNWrapper, self).__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.start_level = start_level
        self.end_level = end_level
        assert start_level >= 0 and end_level >= start_level
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.cat_coors = cat_coors
        self.cat_coors_level = cat_coors_level
        self.fuse_by_cat = fuse_by_cat
        self.return_list = return_list
        self.upsample_times = upsample_times
        self.with_pred = with_pred
        if positional_encoding is not None:
            self.positional_encoding = build_positional_encoding(
                positional_encoding)
        else:
            self.positional_encoding = None

        self.convs_all_levels = nn.ModuleList()
        for i in range(self.start_level, self.end_level + 1):
            convs_per_level = nn.Sequential()
            if i == 0:
                if i == self.cat_coors_level and self.cat_coors:
                    chn = self.in_channels + 2
                else:
                    chn = self.in_channels
                if upsample_times == self.end_level - i:
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(i), one_conv)
                else:
                    for i in range(self.end_level - upsample_times):
                        one_conv = ConvModule(
                            chn,
                            self.feat_channels,
                            3,
                            padding=1,
                            stride=2,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg,
                            inplace=False)
                        convs_per_level.add_module('conv' + str(i), one_conv)
                self.convs_all_levels.append(convs_per_level)
                continue

            for j in range(i):
                if j == 0:
                    if i == self.cat_coors_level and self.cat_coors:
                        chn = self.in_channels + 2
                    else:
                        chn = self.in_channels
                    one_conv = ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                        inplace=False)
                    convs_per_level.add_module('conv' + str(j), one_conv)
                    if j < upsample_times - (self.end_level - i):
                        one_upsample = nn.Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=False)
                        convs_per_level.add_module('upsample' + str(j),
                                                   one_upsample)
                    continue

                one_conv = ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    inplace=False)
                convs_per_level.add_module('conv' + str(j), one_conv)
                if j < upsample_times - (self.end_level - i):
                    one_upsample = nn.Upsample(
                        scale_factor=2, mode='bilinear', align_corners=False)
                    convs_per_level.add_module('upsample' + str(j),
                                               one_upsample)

            self.convs_all_levels.append(convs_per_level)

        if fuse_by_cat:
            in_channels = self.feat_channels * len(self.convs_all_levels)
        else:
            in_channels = self.feat_channels

        if self.with_pred:
            self.conv_pred = ConvModule(
                in_channels,
                self.out_channels,
                1,
                padding=0,
                conv_cfg=self.conv_cfg,
                act_cfg=out_act_cfg,
                norm_cfg=self.norm_cfg)

        self.num_aux_convs = num_aux_convs
        self.aux_convs = nn.ModuleList()
        for i in range(num_aux_convs):
            self.aux_convs.append(
                ConvModule(
                    in_channels,
                    self.out_channels,
                    1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    act_cfg=out_act_cfg,
                    norm_cfg=self.norm_cfg))

    def generate_coord(self, input_feat):
        x_range = torch.linspace(
            -1, 1, input_feat.shape[-1], device=input_feat.device)
        y_range = torch.linspace(
            -1, 1, input_feat.shape[-2], device=input_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([input_feat.shape[0], 1, -1, -1])
        x = x.expand([input_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        return coord_feat

    def forward(self, inputs):
        mlvl_feats = []
        for i in range(self.start_level, self.end_level + 1):
            input_p = inputs[i]
            if i == self.cat_coors_level:
                if self.positional_encoding is not None:
                    ignore_mask = input_p.new_zeros(
                        (input_p.shape[0], input_p.shape[-2],
                         input_p.shape[-1]),
                        dtype=torch.bool)
                    positional_encoding = self.positional_encoding(ignore_mask)
                    input_p = input_p + positional_encoding
                if self.cat_coors:
                    coord_feat = self.generate_coord(input_p)
                    input_p = torch.cat([input_p, coord_feat], 1)

            mlvl_feats.append(self.convs_all_levels[i](input_p))

        if self.fuse_by_cat:
            feature_add_all_level = torch.cat(mlvl_feats, dim=1)
        else:
            feature_add_all_level = sum(mlvl_feats)

        if self.with_pred:
            out = self.conv_pred(feature_add_all_level)
        else:
            out = feature_add_all_level

        if self.num_aux_convs > 0:
            outs = [out]
            for conv in self.aux_convs:
                outs.append(conv(feature_add_all_level))
            return outs

        if self.return_list:
            return [out]
        else:
            return out
