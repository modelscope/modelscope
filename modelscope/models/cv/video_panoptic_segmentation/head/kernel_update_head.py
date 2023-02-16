# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, bias_init_with_prob, build_activation_layer,
                      build_norm_layer)
from mmcv.cnn.bricks.transformer import (FFN, MultiheadAttention,
                                         build_transformer_layer)
from mmcv.runner import force_fp32
from mmdet.core import multi_apply
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.atss_head import reduce_mean
from mmdet.models.losses import accuracy
from mmdet.utils import get_root_logger

from .mask import tensor_mask2box


@HEADS.register_module()
class VideoKernelUpdateHead(nn.Module):

    def __init__(
        self,
        num_classes=80,
        num_ffn_fcs=2,
        num_heads=8,
        num_cls_fcs=1,
        num_mask_fcs=3,
        feedforward_channels=2048,
        in_channels=256,
        out_channels=256,
        dropout=0.0,
        mask_thr=0.5,
        act_cfg=dict(type='ReLU', inplace=True),
        ffn_act_cfg=dict(type='ReLU', inplace=True),
        conv_kernel_size=3,
        feat_transform_cfg=None,
        hard_mask_thr=0.5,
        kernel_init=False,
        with_ffn=True,
        mask_out_stride=4,
        relative_coors=False,
        relative_coors_off=False,
        feat_gather_stride=1,
        mask_transform_stride=1,
        mask_upsample_stride=1,
        num_thing_classes=80,
        num_stuff_classes=53,
        mask_assign_stride=4,
        ignore_label=255,
        thing_label_in_seg=0,
        previous=None,
        previous_x_feat=None,
        previous_link=None,  # seg/cls embeddings
        previous_type=None,  # tracking embeddings
        previous_detach=False,
        previous_detach_link=False,  # whether detach linl query
        previous_link_detach=False,
        kernel_updator_cfg=dict(
            type='DynamicConv',
            in_channels=256,
            feat_channels=64,
            out_channels=256,
            input_feat_shape=1,
            act_cfg=dict(type='ReLU', inplace=True),
            norm_cfg=dict(type='LN')),
        loss_rank=None,
        loss_mask=dict(
            type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
        loss_dice=dict(type='DiceLoss', loss_weight=3.0),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0)):
        super(VideoKernelUpdateHead, self).__init__()
        self.num_classes = num_classes
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        if loss_rank is not None:
            self.loss_rank = build_loss(loss_rank)
        else:
            self.loss_rank = loss_rank

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mask_thr = mask_thr
        self.fp16_enabled = False
        self.dropout = dropout

        self.num_heads = num_heads
        self.hard_mask_thr = hard_mask_thr
        self.kernel_init = kernel_init
        self.with_ffn = with_ffn
        self.mask_out_stride = mask_out_stride
        self.relative_coors = relative_coors
        self.relative_coors_off = relative_coors_off
        self.conv_kernel_size = conv_kernel_size
        self.feat_gather_stride = feat_gather_stride
        self.mask_transform_stride = mask_transform_stride
        self.mask_upsample_stride = mask_upsample_stride

        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.ignore_label = ignore_label
        self.thing_label_in_seg = thing_label_in_seg

        self.attention = MultiheadAttention(in_channels * conv_kernel_size**2,
                                            num_heads, dropout)
        self.attention_norm = build_norm_layer(
            dict(type='LN'), in_channels * conv_kernel_size**2)[1]

        self.kernel_update_conv = build_transformer_layer(kernel_updator_cfg)

        if feat_transform_cfg is not None:
            kernel_size = feat_transform_cfg.pop('kernel_size', 1)
            self.feat_transform = ConvModule(
                in_channels,
                in_channels,
                kernel_size,
                stride=feat_gather_stride,
                padding=int(feat_gather_stride // 2),
                **feat_transform_cfg)
        else:
            self.feat_transform = None

        if self.with_ffn:
            self.ffn = FFN(
                in_channels,
                feedforward_channels,
                num_ffn_fcs,
                act_cfg=ffn_act_cfg,
                dropout=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.cls_fcs = nn.ModuleList()
        for _ in range(num_cls_fcs):
            self.cls_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.cls_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.cls_fcs.append(build_activation_layer(act_cfg))

        if self.loss_cls.use_sigmoid:
            self.fc_cls = nn.Linear(in_channels, self.num_classes)
        else:
            self.fc_cls = nn.Linear(in_channels, self.num_classes + 1)

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

        self.previous = previous
        self.previous_type = previous_type
        self.previous_link = previous_link
        self.previous_x_feat = previous_x_feat
        self.previous_detach = previous_detach
        self.previous_detach_link = previous_detach_link
        self.previous_link_detach = previous_link_detach

        if self.previous is not None:
            _in_channels = self.in_channels
            _conv_kernel_size = self.conv_kernel_size
            _num_head = 8
            _dropout = 0.
            # tracking embedding
            if self.previous_type == 'ffn':
                self.attention_previous = MultiheadAttention(
                    _in_channels * _conv_kernel_size**2,
                    _num_head,
                    _dropout,
                )
                _, self.attention_previous_norm = build_norm_layer(
                    dict(type='LN'), _in_channels * _conv_kernel_size**2)
                # add link ffn
                self.link_ffn = FFN(
                    in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    dropout=dropout)
                self.link_ffn_norm = build_norm_layer(
                    dict(type='LN'), in_channels)[1]

            elif self.previous_type == 'update' or self.previous_type == 'update_obj':

                self.attention_previous_update_track = build_transformer_layer(
                    kernel_updator_cfg)

                self.attention_previous_track = MultiheadAttention(
                    _in_channels * _conv_kernel_size**2,
                    _num_head,
                    _dropout,
                )
                _, self.attention_previous_norm_track = build_norm_layer(
                    dict(type='LN'), _in_channels * _conv_kernel_size**2)
                # add link ffn
                self.link_ffn_track = FFN(
                    in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    dropout=dropout)
                self.link_ffn_norm_track = build_norm_layer(
                    dict(type='LN'), in_channels)[1]

            # seg and cls embedding Link
            if self.previous_link == 'update_dynamic_cov':
                _in_channels = self.in_channels
                _conv_kernel_size = self.conv_kernel_size
                _num_head = 8
                _dropout = 0.
                self.attention_previous_update_link = build_transformer_layer(
                    kernel_updator_cfg)
                self.attention_previous_link = MultiheadAttention(
                    _in_channels * _conv_kernel_size**2,
                    _num_head,
                    _dropout,
                )
                _, self.attention_previous_norm_link = build_norm_layer(
                    dict(type='LN'), _in_channels * _conv_kernel_size**2)
                # add link ffn
                self.link_ffn_link = FFN(
                    in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    dropout=dropout)
                self.link_ffn_norm_link = build_norm_layer(
                    dict(type='LN'), in_channels)[1]

            elif self.previous_link == 'link_atten':
                _in_channels = self.in_channels
                _conv_kernel_size = self.conv_kernel_size
                _num_head = 8
                _dropout = 0.
                self.attention_previous_link = MultiheadAttention(
                    _in_channels * _conv_kernel_size**2,
                    _num_head,
                    _dropout,
                )
                _, self.attention_previous_norm_link = build_norm_layer(
                    dict(type='LN'), _in_channels * _conv_kernel_size**2)
                # add link ffn
                self.link_ffn_link = FFN(
                    in_channels,
                    feedforward_channels,
                    num_ffn_fcs,
                    act_cfg=ffn_act_cfg,
                    dropout=dropout)
                self.link_ffn_norm_link = build_norm_layer(
                    dict(type='LN'), in_channels)[1]

    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None,
                previous_obj_feats=None,
                previous_mask_preds=None,
                previous_x_feats=None):

        N, num_proposals = proposal_feat.shape[:2]
        if self.feat_transform is not None:
            x = self.feat_transform(x)
            if previous_x_feats is not None:
                previous_x_feats = self.feat_transform(previous_x_feats)
        C, H, W = x.shape[-3:]

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds, (H, W), align_corners=False, mode='bilinear')
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        x_feat = torch.einsum('bnhw,bchw->bnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        proposal_feat = proposal_feat.reshape(N, num_proposals,
                                              self.in_channels,
                                              -1).permute(0, 1, 3, 2)

        # whether to detach the previous outputs
        if self.training and self.previous_detach:
            previous_obj_feats = previous_obj_feats.detach()

        # update previous with link object query
        if previous_obj_feats is not None and self.previous_link == 'update_dynamic_cov':
            previous_obj_feats_link = previous_obj_feats.reshape(
                N, num_proposals, self.in_channels, -1).permute(0, 1, 3, 2)

            if self.training and self.previous_detach_link:
                previous_obj_feats_link = previous_obj_feats_link.detach()

            previous_obj_feats_update = self.attention_previous_update_link(
                x_feat, previous_obj_feats_link)

            previous_obj_feats_update = previous_obj_feats_update.reshape(
                N, num_proposals, -1).permute(1, 0, 2)
            cur_obj_feat = proposal_feat.reshape(N, num_proposals, self.in_channels * self.conv_kernel_size ** 2). \
                permute(1, 0, 2)
            cur_obj_feat = self.attention_previous_norm_link(
                self.attention_previous_link(
                    query=cur_obj_feat,
                    key=previous_obj_feats_update,
                    value=previous_obj_feats_update,
                    identity=cur_obj_feat), )
            cur_obj_feat = cur_obj_feat.permute(1, 0, 2)
            cur_obj_feat = cur_obj_feat.reshape(N, num_proposals, -1,
                                                self.in_channels)
            # pre_obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
            proposal_feat = self.link_ffn_norm_link(
                self.link_ffn_link(cur_obj_feat))

        if previous_obj_feats is not None and self.previous_link == 'link_atten':
            previous_obj_feats_link = previous_obj_feats.reshape(
                N, num_proposals, self.in_channels, -1).permute(0, 1, 3, 2)

            previous_obj_feats_update = previous_obj_feats_link.reshape(
                N, num_proposals, -1).permute(1, 0, 2)
            cur_obj_feat = proposal_feat.reshape(N, num_proposals, self.in_channels * self.conv_kernel_size ** 2). \
                permute(1, 0, 2)
            cur_obj_feat = self.attention_previous_norm_link(
                self.attention_previous_link(
                    query=cur_obj_feat,
                    key=previous_obj_feats_update,
                    value=previous_obj_feats_update,
                    identity=cur_obj_feat), )
            cur_obj_feat = cur_obj_feat.permute(1, 0, 2)
            cur_obj_feat = cur_obj_feat.reshape(N, num_proposals, -1,
                                                self.in_channels)
            # pre_obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
            proposal_feat = self.link_ffn_norm_link(
                self.link_ffn_link(cur_obj_feat))

        # update current
        obj_feat = self.kernel_update_conv(x_feat, proposal_feat)

        # [B, N, K*K, C] -> [B, N, K*K*C] -> [N, B, K*K*C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1).permute(1, 0, 2)
        obj_feat = self.attention_norm(self.attention(obj_feat))
        # [N, B, K*K*C] -> [B, N, K*K*C]
        obj_feat = obj_feat.permute(1, 0, 2)

        # obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
        obj_feat = obj_feat.reshape(N, num_proposals, -1, self.in_channels)

        # FFN
        if self.with_ffn:
            obj_feat = self.ffn_norm(self.ffn(obj_feat))

        # For Tracking Parts
        # Link previous and cur if previous obj feat is Not None
        if previous_obj_feats is not None:
            # previous_obj_feats (b, n, c, k, k) -> (b,n,c,k*k) -> (b,,n, k*k, c)
            # permute to correct dimension

            if self.previous_type == 'ffn':
                previous_obj_feats = previous_obj_feats.reshape(
                    N, num_proposals, self.in_channels,
                    -1).permute(0, 1, 3, 2)
                cur_obj_feat = obj_feat.reshape(N, num_proposals, self.in_channels * self.conv_kernel_size ** 2). \
                    permute(1, 0, 2)
                previous_obj_feats = previous_obj_feats.reshape(
                    N, num_proposals,
                    self.in_channels * self.conv_kernel_size**2).permute(
                        1, 0, 2)

                previous_obj_feat = self.attention_previous_norm(
                    self.attention_previous(
                        query=cur_obj_feat,
                        key=previous_obj_feats,
                        value=previous_obj_feats,
                        identity=cur_obj_feat), )
                previous_obj_feat = previous_obj_feat.permute(1, 0, 2)
                previous_obj_feat_track = previous_obj_feat.reshape(
                    N, num_proposals, -1, self.in_channels)
                # pre_obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
                previous_obj_feat_track = self.link_ffn_norm(
                    self.link_ffn(previous_obj_feat_track))

            elif self.previous_type == 'update':
                # not work
                previous_obj_feats = previous_obj_feats.reshape(
                    N, num_proposals, self.in_channels,
                    -1).permute(0, 1, 3, 2)
                previous_obj_feats_track = self.attention_previous_update_track(
                    x_feat, previous_obj_feats)

                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals, self.in_channels,
                    -1).permute(0, 1, 3, 2)
                cur_obj_feat = obj_feat.reshape(N, num_proposals, self.in_channels * self.conv_kernel_size ** 2). \
                    permute(1, 0, 2)
                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals,
                    self.in_channels * self.conv_kernel_size**2).permute(
                        1, 0, 2)

                previous_obj_feats_track = self.attention_previous_norm_track(
                    self.attention_previous_track(
                        query=cur_obj_feat,
                        key=previous_obj_feats_track,
                        value=previous_obj_feats_track,
                        identity=cur_obj_feat), )
                previous_obj_feats_track = previous_obj_feats_track.permute(
                    1, 0, 2)
                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals, -1, self.in_channels)
                # pre_obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
                previous_obj_feat_track = self.link_ffn_norm_track(
                    self.link_ffn_track(previous_obj_feats_track))

            elif self.previous_type == 'update_obj':
                # not work
                previous_obj_feats = previous_obj_feats.reshape(
                    N, num_proposals, self.in_channels,
                    -1).permute(0, 1, 3, 2)
                previous_obj_feats_track = self.attention_previous_update_track(
                    obj_feat.squeeze(2), previous_obj_feats)

                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals, self.in_channels,
                    -1).permute(0, 1, 3, 2)
                cur_obj_feat = obj_feat.reshape(N, num_proposals, self.in_channels * self.conv_kernel_size ** 2). \
                    permute(1, 0, 2)
                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals,
                    self.in_channels * self.conv_kernel_size**2).permute(
                        1, 0, 2)

                previous_obj_feats_track = self.attention_previous_norm_track(
                    self.attention_previous_track(
                        query=cur_obj_feat,
                        key=previous_obj_feats_track,
                        value=previous_obj_feats_track,
                        identity=cur_obj_feat), )
                previous_obj_feats_track = previous_obj_feats_track.permute(
                    1, 0, 2)
                previous_obj_feats_track = previous_obj_feats_track.reshape(
                    N, num_proposals, -1, self.in_channels)
                # pre_obj_feat in shape [B, N, K*K*C] -> [B, N, K*K, C]
                previous_obj_feat_track = self.link_ffn_norm_track(
                    self.link_ffn_track(previous_obj_feats_track))
            else:
                previous_obj_feat_track = None

        cls_feat = obj_feat.sum(-2)
        mask_feat = obj_feat

        for cls_layer in self.cls_fcs:
            cls_feat = cls_layer(cls_feat)
        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)

        cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
        else:
            mask_x = x
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        new_mask_preds = []
        for i in range(N):
            new_mask_preds.append(
                F.conv2d(
                    mask_x[i:i + 1],
                    mask_feat[i],
                    padding=int(self.conv_kernel_size // 2)))

        new_mask_preds = torch.cat(new_mask_preds, dim=0)
        new_mask_preds = new_mask_preds.reshape(N, num_proposals, H, W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')

        if previous_obj_feats is not None and previous_obj_feat_track is not None:
            obj_feat = obj_feat.permute(0, 1, 3,
                                        2).reshape(N, num_proposals,
                                                   self.in_channels,
                                                   self.conv_kernel_size,
                                                   self.conv_kernel_size)
            previous_obj_feat_track = previous_obj_feat_track.permute(
                0, 1, 3,
                2).reshape(N, num_proposals, self.in_channels,
                           self.conv_kernel_size, self.conv_kernel_size)
            return cls_score, new_mask_preds, obj_feat, x_feat, previous_obj_feat_track
        else:
            return cls_score, new_mask_preds, obj_feat.permute(
                0, 1, 3, 2).reshape(N, num_proposals, self.in_channels,
                                    self.conv_kernel_size,
                                    self.conv_kernel_size), x_feat, None

    def _get_target_single(self, pos_inds, neg_inds, pos_mask, neg_mask,
                           pos_gt_mask, pos_gt_labels, gt_sem_seg, gt_sem_cls,
                           cfg):

        num_pos = pos_mask.size(0)
        num_neg = neg_mask.size(0)
        num_samples = num_pos + num_neg
        H, W = pos_mask.shape[-2:]
        # original implementation uses new_zeros since BG are set to be 0
        # now use empty & fill because BG cat_id = num_classes,
        # FG cat_id = [0, num_classes-1]
        labels = pos_mask.new_full((num_samples, ),
                                   self.num_classes,
                                   dtype=torch.long)
        label_weights = pos_mask.new_zeros((num_samples, self.num_classes))
        mask_targets = pos_mask.new_zeros(num_samples, H, W)
        mask_weights = pos_mask.new_zeros(num_samples, H, W)
        if num_pos > 0:
            labels[pos_inds] = pos_gt_labels
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            label_weights[pos_inds] = pos_weight
            pos_mask_targets = pos_gt_mask
            mask_targets[pos_inds, ...] = pos_mask_targets
            mask_weights[pos_inds, ...] = 1

        if num_neg > 0:
            label_weights[neg_inds] = 1.0

        if gt_sem_cls is not None and gt_sem_seg is not None:
            sem_labels = pos_mask.new_full((self.num_stuff_classes, ),
                                           self.num_classes,
                                           dtype=torch.long)
            sem_targets = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_weights = pos_mask.new_zeros(self.num_stuff_classes, H, W)
            sem_stuff_weights = torch.eye(
                self.num_stuff_classes, device=pos_mask.device)
            sem_thing_weights = pos_mask.new_zeros(
                (self.num_stuff_classes, self.num_thing_classes))
            sem_label_weights = torch.cat(
                [sem_thing_weights, sem_stuff_weights], dim=-1)
            if len(gt_sem_cls > 0):
                sem_inds = gt_sem_cls - self.num_thing_classes
                sem_inds = sem_inds.long()
                sem_labels[sem_inds] = gt_sem_cls.long()
                sem_targets[sem_inds] = gt_sem_seg
                sem_weights[sem_inds] = 1

            label_weights[:, self.num_thing_classes:] = 0
            labels = torch.cat([labels, sem_labels])
            label_weights = torch.cat([label_weights, sem_label_weights])
            mask_targets = torch.cat([mask_targets, sem_targets])
            mask_weights = torch.cat([mask_weights, sem_weights])

        return labels, label_weights, mask_targets, mask_weights

    def get_targets(self,
                    sampling_results,
                    gt_mask,
                    gt_labels,
                    rcnn_train_cfg,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None):

        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_sem_seg is None:
            gt_sem_seg = [None] * 2
            gt_sem_cls = [None] * 2

        labels, label_weights, mask_targets, mask_weights = multi_apply(
            self._get_target_single,
            pos_inds_list,
            neg_inds_list,
            pos_mask_list,
            neg_mask_list,
            pos_gt_mask_list,
            pos_gt_labels_list,
            gt_sem_seg,
            gt_sem_cls,
            cfg=rcnn_train_cfg)
        if concat:
            labels = torch.cat(labels, 0)
            label_weights = torch.cat(label_weights, 0)
            mask_targets = torch.cat(mask_targets, 0)
            mask_weights = torch.cat(mask_weights, 0)
        return labels, label_weights, mask_targets, mask_weights

    def rescale_masks(self, masks_per_img, img_meta):
        h, w, _ = img_meta['img_shape']
        masks_per_img = F.interpolate(
            masks_per_img.unsqueeze(0).sigmoid(),
            size=img_meta['batch_input_shape'],
            mode='bilinear',
            align_corners=False)

        masks_per_img = masks_per_img[:, :, :h, :w]
        ori_shape = img_meta['ori_shape']
        seg_masks = F.interpolate(
            masks_per_img,
            size=ori_shape[:2],
            mode='bilinear',
            align_corners=False).squeeze(0)
        return seg_masks

    def get_seg_masks(self, masks_per_img, labels_per_img, scores_per_img,
                      test_cfg, img_meta):
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg.mask_thr
        bbox_result, segm_result, mask_preds = self.segm2result(
            seg_masks, labels_per_img, scores_per_img)
        return bbox_result, segm_result, mask_preds

    def segm2result(self, mask_preds, det_labels, cls_scores):
        num_classes = self.num_classes
        # bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes mask to bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bboxes[:, :4] = np.array(tensor_mask2box(mask_preds).clip(min=0))

        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bboxes, segm_result, mask_preds
