# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license

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

from ..utils import outs2results


@HEADS.register_module()
class KernelUpdateHeadVideo(nn.Module):

    def __init__(
        self,
        with_cls=True,
        num_proposals=100,
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
        # query fusion
        query_merge_method='mean',
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
        super().__init__()
        self.num_proposals = num_proposals
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
                ffn_drop=dropout)
            self.ffn_norm = build_norm_layer(dict(type='LN'), in_channels)[1]

        self.with_cls = with_cls
        if self.with_cls:
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

        # query fusion
        self.query_merge_method = query_merge_method
        if self.query_merge_method == 'attention' and self.with_cls:
            _num_head = 8
            _drop_out = 0.
            self.query_merge_attn = MultiheadAttention(
                self.in_channels, _num_head, _drop_out, batch_first=True)
            self.query_merge_norm = build_norm_layer(
                dict(type='LN'), self.in_channels)[1]
            self.query_merge_ffn = FFN(
                self.in_channels,
                self.in_channels * 8,
                num_ffn_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.)
            self.query_merge_ffn_norm = build_norm_layer(
                dict(type='LN'), self.in_channels)[1]
        elif self.query_merge_method == 'attention_pos' and self.with_cls:
            _num_head = 8
            _drop_out = 0.
            self.query_merge_attn = MultiheadAttention(
                self.in_channels, _num_head, _drop_out, batch_first=True)
            self.query_merge_norm = build_norm_layer(
                dict(type='LN'), self.in_channels)[1]
            self.query_merge_ffn = FFN(
                self.in_channels,
                self.in_channels * 8,
                num_ffn_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.)
            self.query_merge_ffn_norm = build_norm_layer(
                dict(type='LN'), self.in_channels)[1]

        self.mask_fcs = nn.ModuleList()
        for _ in range(num_mask_fcs):
            self.mask_fcs.append(
                nn.Linear(in_channels, in_channels, bias=False))
            self.mask_fcs.append(
                build_norm_layer(dict(type='LN'), in_channels)[1])
            self.mask_fcs.append(build_activation_layer(act_cfg))

        self.fc_mask = nn.Linear(in_channels, out_channels)

    def init_weights(self):
        """Use xavier initialization for all weight parameter and set
        classification head bias as a specific value when use focal loss."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                # adopt the default initialization for
                # the weight and bias of the layer norm
                pass
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            nn.init.constant_(self.fc_cls.bias, bias_init)
        if self.kernel_init:
            logger = get_root_logger()
            logger.info(
                'mask kernel in mask head is normal initialized by std 0.01')
            nn.init.normal_(self.fc_mask.weight, mean=0, std=0.01)

    def forward(self,
                x,
                proposal_feat,
                mask_preds,
                prev_cls_score=None,
                mask_shape=None,
                img_metas=None,
                pos=None):
        if len(proposal_feat.size()) == 6:
            assert not self.with_cls
            is_gather_query = False
            N, _, num_proposals = proposal_feat.shape[:3]
        else:
            assert self.with_cls
            is_gather_query = True
            N, num_proposals = proposal_feat.shape[:2]
        assert self.num_proposals == num_proposals
        _, num_frames, C, H, W = x.size()
        if self.feat_transform is not None:
            x = self.feat_transform(x.reshape(
                (N * num_frames, C, H, W))).reshape((N, num_frames, C, H, W))

        mask_h, mask_w = mask_preds.shape[-2:]
        if mask_h != H or mask_w != W:
            gather_mask = F.interpolate(
                mask_preds.reshape((N * num_proposals, C, H, W)), (H, W),
                align_corners=False,
                mode='bilinear').reshape((N, num_frames, C, H, W))
        else:
            gather_mask = mask_preds

        sigmoid_masks = gather_mask.sigmoid()
        nonzero_inds = sigmoid_masks > self.hard_mask_thr
        sigmoid_masks = nonzero_inds.float()

        # einsum is faster than bmm by 30%
        if is_gather_query:
            # x_feat = torch.einsum('bfnhw,bfchw->bnc', sigmoid_masks, x)
            if self.query_merge_method == 'mean':
                x_feat = torch.einsum('bfnhw,bfchw->bfnc', sigmoid_masks,
                                      x).mean(1)
            elif self.query_merge_method == 'attention':
                x_feat = torch.einsum('bfnhw,bfchw->bfnc', sigmoid_masks, x)
                x_feat = x_feat.reshape(
                    (N, num_frames * num_proposals, self.in_channels))
                assert proposal_feat.size()[-2:] == (
                    1, 1), 'Only supporting kernel size = 1'
                init_query = proposal_feat.reshape(N, num_proposals,
                                                   self.in_channels).detach()
                x_feat = self.query_merge_attn(
                    query=init_query, key=x_feat, value=x_feat)
                x_feat = self.query_merge_norm(x_feat)
                x_feat = self.query_merge_ffn_norm(
                    self.query_merge_ffn(x_feat))
            elif self.query_merge_method == 'attention_pos':
                x_feat = torch.einsum('bfnhw,bfchw->bfnc', sigmoid_masks, x)
                x_feat = x_feat.reshape(
                    (N, num_frames * num_proposals, self.in_channels))
                assert proposal_feat.size()[-2:] == (
                    1, 1), 'Only supporting kernel size = 1'
                init_query = proposal_feat.reshape(N, num_proposals,
                                                   self.in_channels).detach()
                query_pos = pos.repeat(N, 1, 1)
                key_pos = query_pos.repeat(1, num_frames, 1)
                x_feat = self.query_merge_attn(
                    query=init_query,
                    key=x_feat,
                    value=x_feat,
                    query_pos=query_pos,
                    key_pos=key_pos)
                x_feat = self.query_merge_norm(x_feat)
                x_feat = self.query_merge_ffn_norm(
                    self.query_merge_ffn(x_feat))
            else:
                raise NotImplementedError
        else:
            x_feat = torch.einsum('bfnhw,bfchw->bfnc', sigmoid_masks, x)

        # obj_feat in shape [B, N, C, K, K] -> [B, N, C, K*K] -> [B, N, K*K, C]
        if is_gather_query:
            proposal_feat = proposal_feat.reshape(N, num_proposals,
                                                  self.in_channels,
                                                  -1).permute(0, 1, 3, 2)
            obj_feat = self.kernel_update_conv(x_feat, proposal_feat)
        else:
            proposal_feat = proposal_feat.reshape(N * num_frames,
                                                  num_proposals,
                                                  self.in_channels,
                                                  -1).permute(0, 1, 3, 2)
            obj_feat = self.kernel_update_conv(
                x_feat.reshape(N * num_frames, num_proposals, C),
                proposal_feat)
            N *= num_frames

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

        mask_feat = obj_feat

        if is_gather_query:
            cls_feat = obj_feat.sum(-2)
            for cls_layer in self.cls_fcs:
                cls_feat = cls_layer(cls_feat)
            cls_score = self.fc_cls(cls_feat).view(N, num_proposals, -1)
        else:
            cls_score = None

        for reg_layer in self.mask_fcs:
            mask_feat = reg_layer(mask_feat)
        # [B, N, K*K, C] -> [B, N, C, K*K]
        mask_feat = self.fc_mask(mask_feat).permute(0, 1, 3, 2)

        if (self.mask_transform_stride == 2 and self.feat_gather_stride == 1):
            mask_x = F.interpolate(
                x, scale_factor=0.5, mode='bilinear', align_corners=False)
            H, W = mask_x.shape[-2:]
            raise NotImplementedError
        else:
            mask_x = x
        # group conv is 5x faster than unfold and uses about 1/5 memory
        # Group conv vs. unfold vs. concat batch, 2.9ms :13.5ms :3.8ms
        # Group conv vs. unfold vs. concat batch, 278 : 1420 : 369
        # fold_x = F.unfold(
        #     mask_x,
        #     self.conv_kernel_size,
        #     padding=int(self.conv_kernel_size // 2))
        # mask_feat = mask_feat.reshape(N, num_proposals, -1)
        # new_mask_preds = torch.einsum('bnc,bcl->bnl', mask_feat, fold_x)
        # [B, N, C, K*K] -> [B*N, C, K, K]
        mask_feat = mask_feat.reshape(N, num_proposals, C,
                                      self.conv_kernel_size,
                                      self.conv_kernel_size)
        # [B, C, H, W] -> [1, B*C, H, W]
        if is_gather_query:
            new_mask_preds = []
            for i in range(N):
                new_mask_preds.append(
                    F.conv2d(
                        mask_x[i],
                        mask_feat[i],
                        padding=int(self.conv_kernel_size // 2)))

            new_mask_preds = torch.stack(new_mask_preds, dim=0)
            assert new_mask_preds.size() == (N, num_frames, num_proposals, H,
                                             W)
        else:
            N = N // num_frames
            new_mask_preds = []
            for i in range(N):
                for j in range(num_frames):
                    new_mask_preds.append(
                        F.conv2d(
                            mask_x[i][j][None],
                            mask_feat[i * num_frames + j],
                            padding=int(self.conv_kernel_size // 2)))
            new_mask_preds = torch.cat(new_mask_preds, dim=0)
            new_mask_preds = new_mask_preds.reshape(N, num_frames,
                                                    num_proposals, H, W)
            assert new_mask_preds.size() == (N, num_frames, num_proposals, H,
                                             W)
        if self.mask_transform_stride == 2:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                scale_factor=2,
                mode='bilinear',
                align_corners=False)
            raise NotImplementedError

        if mask_shape is not None and mask_shape[0] != H:
            new_mask_preds = F.interpolate(
                new_mask_preds,
                mask_shape,
                align_corners=False,
                mode='bilinear')
            raise NotImplementedError
        if is_gather_query:
            return cls_score, new_mask_preds, obj_feat.permute(
                0, 1, 3, 2).reshape(N, num_proposals, self.in_channels,
                                    self.conv_kernel_size,
                                    self.conv_kernel_size)
        else:
            return None, new_mask_preds, obj_feat.permute(0, 1, 3, 2).reshape(
                N, num_frames, num_proposals, self.in_channels,
                self.conv_kernel_size, self.conv_kernel_size)

    @force_fp32(apply_to=('cls_score', 'mask_pred'))
    def loss(self,
             object_feats,
             cls_score,
             mask_pred,
             labels,
             label_weights,
             mask_targets,
             mask_weights,
             imgs_whwh=None,
             reduction_override=None,
             **kwargs):

        losses = dict()
        bg_class_ind = self.num_classes
        # note in spare rcnn num_gt == num_pos
        pos_inds = (labels >= 0) & (labels < bg_class_ind)
        num_pos = pos_inds.sum().float()
        avg_factor = reduce_mean(num_pos).clamp_(min=1.0)

        num_preds = mask_pred.shape[0] * mask_pred.shape[1]
        if cls_score is not None:
            assert mask_pred.shape[0] == cls_score.shape[0]
            assert mask_pred.shape[1] == cls_score.shape[1]

        if cls_score is not None:
            if cls_score.numel() > 0:
                losses['loss_cls'] = self.loss_cls(
                    cls_score.view(num_preds, -1),
                    labels,
                    label_weights,
                    avg_factor=avg_factor,
                    reduction_override=reduction_override)
                losses['pos_acc'] = accuracy(
                    cls_score.view(num_preds, -1)[pos_inds], labels[pos_inds])
        if mask_pred is not None:
            bool_pos_inds = pos_inds.type(torch.bool)
            # 0~self.num_classes-1 are FG, self.num_classes is BG
            # do not perform bounding box regression for BG anymore.
            H, W = mask_pred.shape[-2:]
            if pos_inds.any():
                pos_mask_pred = mask_pred.reshape(num_preds, H,
                                                  W)[bool_pos_inds]
                pos_mask_targets = mask_targets[bool_pos_inds]
                losses['loss_mask'] = self.loss_mask(pos_mask_pred,
                                                     pos_mask_targets)
                losses['loss_dice'] = self.loss_dice(pos_mask_pred,
                                                     pos_mask_targets)

                if self.loss_rank is not None:
                    batch_size = mask_pred.size(0)
                    rank_target = mask_targets.new_full((batch_size, H, W),
                                                        self.ignore_label,
                                                        dtype=torch.long)
                    rank_inds = pos_inds.view(batch_size,
                                              -1).nonzero(as_tuple=False)
                    batch_mask_targets = mask_targets.view(
                        batch_size, -1, H, W).bool()
                    for i in range(batch_size):
                        curr_inds = (rank_inds[:, 0] == i)
                        curr_rank = rank_inds[:, 1][curr_inds]
                        for j in curr_rank:
                            rank_target[i][batch_mask_targets[i][j]] = j
                    losses['loss_rank'] = self.loss_rank(
                        mask_pred, rank_target, ignore_index=self.ignore_label)
            else:
                losses['loss_mask'] = mask_pred.sum() * 0
                losses['loss_dice'] = mask_pred.sum() * 0
                if self.loss_rank is not None:
                    losses['loss_rank'] = mask_pred.sum() * 0

        return losses

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
                    rcnn_train_cfg,
                    concat=True,
                    gt_sem_seg=None,
                    gt_sem_cls=None):
        num_imgs = len(sampling_results)
        pos_inds_list = [res.pos_inds for res in sampling_results]
        neg_inds_list = [res.neg_inds for res in sampling_results]
        pos_mask_list = [res.pos_masks for res in sampling_results]
        neg_mask_list = [res.neg_masks for res in sampling_results]
        pos_gt_mask_list = [res.pos_gt_masks for res in sampling_results]
        pos_gt_labels_list = [res.pos_gt_labels for res in sampling_results]
        if gt_sem_seg is None:
            gt_sem_seg = [None] * num_imgs
            gt_sem_cls = [None] * num_imgs

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
        bbox_result, segm_result = self.segm2result(seg_masks, labels_per_img,
                                                    scores_per_img)
        return bbox_result, segm_result

    def segm2result(self, mask_preds, det_labels, cls_scores):
        num_classes = self.num_classes
        bbox_result = None
        segm_result = [[] for _ in range(num_classes)]
        mask_preds = mask_preds.cpu().numpy()
        det_labels = det_labels.cpu().numpy()
        cls_scores = cls_scores.cpu().numpy()
        num_ins = mask_preds.shape[0]
        # fake bboxes
        bboxes = np.zeros((num_ins, 5), dtype=np.float32)
        bboxes[:, -1] = cls_scores
        bbox_result = [bboxes[det_labels == i, :] for i in range(num_classes)]
        for idx in range(num_ins):
            segm_result[det_labels[idx]].append(mask_preds[idx])
        return bbox_result, segm_result

    def get_seg_masks_tracking(self, masks_per_img, labels_per_img,
                               scores_per_img, ids_per_img, test_cfg,
                               img_meta):
        num_ins = masks_per_img.shape[0]
        # resize mask predictions back
        seg_masks = self.rescale_masks(masks_per_img, img_meta)
        seg_masks = seg_masks > test_cfg['mask_thr']
        # fake bboxes
        bboxes = torch.zeros((num_ins, 5), dtype=torch.float32)
        bboxes[:, -1] = scores_per_img
        tracks = outs2results(
            bboxes=bboxes,
            labels=labels_per_img,
            masks=seg_masks,
            ids=ids_per_img,
            num_classes=self.num_classes,
        )
        return tracks['bbox_results'], tracks['mask_results']
