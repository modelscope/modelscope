# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet.core import build_assigner, build_sampler
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead
from mmdet.utils import get_root_logger


@HEADS.register_module()
class KernelFrameIterHeadVideo(BaseRoIHead):

    def __init__(self,
                 mask_head=None,
                 with_mask_init=False,
                 num_stages=3,
                 stage_loss_weights=(1, 1, 1),
                 proposal_feature_channel=256,
                 assign_stages=5,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 query_merge_method='mean',
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 **kwargs):
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.assign_stages = assign_stages
        self.num_proposals = num_proposals
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.query_merge_method = query_merge_method
        self.proposal_feature_channel = proposal_feature_channel
        super().__init__(
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs)
        if self.query_merge_method == 'attention':
            self.init_query = nn.Embedding(self.num_proposals,
                                           self.proposal_feature_channel)
            _num_head = 8
            _drop_out = 0.
            self.query_merge_attn = MultiheadAttention(
                self.proposal_feature_channel,
                _num_head,
                _drop_out,
                batch_first=True)
            self.query_merge_norm = build_norm_layer(
                dict(type='LN'), self.proposal_feature_channel)[1]
            self.query_merge_ffn = FFN(
                self.proposal_feature_channel,
                self.proposal_feature_channel * 8,
                num_ffn_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.)
            self.query_merge_ffn_norm = build_norm_layer(
                dict(type='LN'), self.proposal_feature_channel)[1]
        elif self.query_merge_method == 'attention_pos':
            self.init_query = nn.Embedding(self.num_proposals,
                                           self.proposal_feature_channel)
            self.query_pos = nn.Embedding(self.num_proposals,
                                          self.proposal_feature_channel)
            _num_head = 8
            _drop_out = 0.
            self.query_merge_attn = MultiheadAttention(
                self.proposal_feature_channel,
                _num_head,
                _drop_out,
                batch_first=True)
            self.query_merge_norm = build_norm_layer(
                dict(type='LN'), self.proposal_feature_channel)[1]
            self.query_merge_ffn = FFN(
                self.proposal_feature_channel,
                self.proposal_feature_channel * 8,
                num_ffn_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True),
                ffn_drop=0.)
            self.query_merge_ffn_norm = build_norm_layer(
                dict(type='LN'), self.proposal_feature_channel)[1]

        self.with_mask_init = with_mask_init
        if self.with_mask_init:
            self.fc_mask = nn.Linear(proposal_feature_channel,
                                     proposal_feature_channel)

        self.logger = get_root_logger()

    def init_mask_head(self, bbox_roi_extractor=None, mask_head=None):
        assert bbox_roi_extractor is None
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for idx, head in enumerate(mask_head):
            head.update(with_cls=(idx < self.assign_stages))
            self.mask_head.append(build_head(head))

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for i in range(self.num_stages):
                self.mask_assigner.append(
                    build_assigner(self.train_cfg['assigner']))
                self.current_stage = i
                self.mask_sampler.append(
                    build_sampler(self.train_cfg['sampler'], context=self))

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        raise NotImplementedError

    def _mask_forward(self, stage, x, object_feats, mask_preds):
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats = mask_head(
            x,
            object_feats,
            mask_preds,
            img_metas=None,
            pos=self.query_pos.weight
            if self.query_merge_method == 'attention_pos' else None)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = [
                F.interpolate(
                    mask_preds[i],
                    scale_factor=mask_head.mask_upsample_stride,
                    align_corners=False,
                    mode='bilinear') for i in range(mask_preds.size(0))
            ]
            scaled_mask_preds = torch.stack(scaled_mask_preds)
        else:
            scaled_mask_preds = mask_preds

        mask_results = dict(
            cls_score=cls_score,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)
        return mask_results

    def _query_fusion(self, obj_feats, num_imgs, num_frames):
        if self.query_merge_method == 'mean':
            object_feats = obj_feats.mean(1)
        elif self.query_merge_method == 'attention':
            assert obj_feats.size()[-2:] == (
                1, 1), 'Only supporting kernel size = 1'
            obj_feats = obj_feats.reshape(
                (num_imgs, num_frames * self.num_proposals,
                 self.proposal_feature_channel))
            init_query = self.init_query.weight.expand(
                num_imgs, *self.init_query.weight.size())
            obj_feats = self.query_merge_attn(
                query=init_query, key=obj_feats, value=obj_feats)
            obj_feats = self.query_merge_norm(obj_feats)
            object_feats = self.query_merge_ffn_norm(
                self.query_merge_ffn(obj_feats))
            object_feats = object_feats[..., None, None]
        elif self.query_merge_method == 'attention_pos':
            assert obj_feats.size()[-2:] == (
                1, 1), 'Only supporting kernel size = 1'
            obj_feats = obj_feats.reshape(
                (num_imgs, num_frames * self.num_proposals,
                 self.proposal_feature_channel))
            init_query = self.init_query.weight.expand(
                num_imgs, *self.init_query.weight.size())
            query_pos = self.query_pos.weight.repeat(num_imgs, 1, 1)
            key_pos = query_pos.repeat(1, num_frames, 1)
            obj_feats = self.query_merge_attn(
                query=init_query,
                key=obj_feats,
                value=obj_feats,
                query_pos=query_pos,
                key_pos=key_pos)
            obj_feats = self.query_merge_norm(obj_feats)
            object_feats = self.query_merge_ffn_norm(
                self.query_merge_ffn(obj_feats))
            object_feats = object_feats[..., None, None]

        return object_feats

    def _mask_init(self, object_feats, x_feats, num_imgs):
        assert object_feats.size()[-2:] == (
            1, 1), 'Only supporting kernel size = 1'
        object_feats = object_feats.flatten(-3, -1)  # BNCKK -> BNC
        mask_feat = self.fc_mask(object_feats)[..., None, None]
        mask_preds = []
        for i in range(num_imgs):
            mask_preds.append(F.conv2d(x_feats[i], mask_feat[i], padding=0))

        mask_preds = torch.stack(mask_preds, dim=0)

        return mask_preds

    def forward_train(self, x, ref_img_metas, cls_scores, masks, obj_feats,
                      ref_gt_masks, ref_gt_labels, ref_gt_instance_ids,
                      **kwargs):
        num_imgs = len(ref_img_metas)
        num_frames = len(ref_img_metas[0])
        if len(obj_feats.size()) == 6:
            object_feats = self._query_fusion(obj_feats, num_imgs, num_frames)
        else:
            object_feats = obj_feats

        all_stage_loss = {}
        if self.with_mask_init:
            mask_preds = self._mask_init(object_feats, x, num_imgs)
            assert self.training
            if self.mask_head[0].mask_upsample_stride > 1:
                scaled_mask_preds = [
                    F.interpolate(
                        mask_preds[i],
                        scale_factor=self.mask_head[0].mask_upsample_stride,
                        align_corners=False,
                        mode='bilinear') for i in range(mask_preds.size(0))
                ]
                scaled_mask_preds = torch.stack(scaled_mask_preds)
            else:
                scaled_mask_preds = mask_preds
            _gt_masks_matches = []
            _assign_results = []
            _sampling_results = []
            _pred_masks_concat = []
            for i in range(num_imgs):
                mask_for_assign = scaled_mask_preds[i][:self.
                                                       num_proposals].detach()
                cls_for_assign = None
                assign_result, gt_masks_match = self.mask_assigner[0].assign(
                    mask_for_assign, cls_for_assign, ref_gt_masks[i],
                    ref_gt_labels[i], ref_gt_instance_ids[i])
                _gt_masks_matches.append(gt_masks_match)
                _assign_results.append(assign_result)
                num_bboxes = scaled_mask_preds.size(2)
                h, w = scaled_mask_preds.shape[-2:]
                pred_masks_match = torch.einsum('fqhw->qfhw',
                                                scaled_mask_preds[i]).reshape(
                                                    (num_bboxes, -1, w))
                sampling_result = self.mask_sampler[0].sample(
                    assign_result, pred_masks_match, gt_masks_match)
                _sampling_results.append(sampling_result)
                _pred_masks_concat.append(pred_masks_match)
            pred_masks_concat = torch.stack(_pred_masks_concat)
            mask_targets = self.mask_head[0].get_targets(
                _sampling_results,
                self.train_cfg,
                True,
                gt_sem_seg=None,
                gt_sem_cls=None)

            single_stage_loss = self.mask_head[0].loss(object_feats, None,
                                                       pred_masks_concat,
                                                       *mask_targets)
            for key, value in single_stage_loss.items():
                all_stage_loss[
                    f'tracker_init_{key}'] = value * self.stage_loss_weights[0]
        else:
            mask_preds = masks

        assign_results = []
        for stage in range(self.num_stages):
            if stage == self.assign_stages:
                object_feats = object_feats[:, None].repeat(
                    1, num_frames, 1, 1, 1, 1)
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']

            prev_mask_preds = scaled_mask_preds.detach()
            prev_cls_score = cls_score.detach(
            ) if cls_score is not None else None

            sampling_results = []
            pred_masks_concat = []
            if stage < self.assign_stages:
                assign_results = []
                gt_masks_matches = []
            for i in range(num_imgs):
                if stage < self.assign_stages:
                    mask_for_assign = prev_mask_preds[i][:, :self.
                                                         num_proposals]
                    if prev_cls_score is not None:
                        cls_for_assign = prev_cls_score[
                            i][:self.num_proposals, :self.num_thing_classes]
                    else:
                        cls_for_assign = None
                    assign_result, gt_masks_match = self.mask_assigner[
                        stage].assign(mask_for_assign, cls_for_assign,
                                      ref_gt_masks[i], ref_gt_labels[i],
                                      ref_gt_instance_ids[i])
                    gt_masks_matches.append(gt_masks_match)
                    assign_results.append(assign_result)
                num_bboxes = scaled_mask_preds.size(2)
                h, w = scaled_mask_preds.shape[-2:]
                pred_masks_match = torch.einsum('fqhw->qfhw',
                                                scaled_mask_preds[i]).reshape(
                                                    (num_bboxes, -1, w))
                sampling_result = self.mask_sampler[stage].sample(
                    assign_results[i], pred_masks_match, gt_masks_matches[i])
                sampling_results.append(sampling_result)
                pred_masks_concat.append(pred_masks_match)
            pred_masks_concat = torch.stack(pred_masks_concat)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                self.train_cfg,
                True,
                gt_sem_seg=None,
                gt_sem_cls=None)

            single_stage_loss = self.mask_head[stage].loss(
                object_feats, cls_score, pred_masks_concat, *mask_targets)
            for key, value in single_stage_loss.items():
                all_stage_loss[
                    f'tracker_s{stage}_{key}'] = value * self.stage_loss_weights[
                        stage]

        features = {
            'obj_feats': object_feats,
            'x_feats': x,
            'cls_scores': cls_score,
            'masks': mask_preds,
        }
        return all_stage_loss, features

    def simple_test(self, x, img_metas, ref_img_metas, cls_scores, masks,
                    obj_feats, **kwargs):
        num_imgs = len(ref_img_metas)
        num_frames = len(ref_img_metas[0])

        if len(obj_feats.size()) == 6:
            object_feats = self._query_fusion(obj_feats, num_imgs, num_frames)
        else:
            object_feats = obj_feats

        if self.with_mask_init:
            mask_preds = self._mask_init(object_feats, x, num_imgs)
        else:
            mask_preds = masks

        cls_score = None
        for stage in range(self.num_stages):
            if stage == self.assign_stages:
                object_feats = object_feats[:, None].repeat(
                    1, num_frames, 1, 1, 1, 1)
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score'] if mask_results[
                'cls_score'] is not None else cls_score
            object_feats = mask_results['object_feats']

        num_classes = self.mask_head[-1].num_classes
        results = []
        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        for img_id in range(num_imgs):
            result = []
            cls_score_per_img = cls_score[img_id]
            # h, quite tricky here, a bounding box can predict multiple results with different labels
            scores_per_img, topk_indices = cls_score_per_img.flatten(
                0, 1).topk(
                    self.test_cfg['max_per_img'], sorted=True)
            mask_indices = topk_indices // num_classes
            # Use the following when torch >= 1.9.0
            # mask_indices = torch.div(topk_indices, num_classes, rounding_mode='floor')
            labels_per_img = topk_indices % num_classes
            for frame_id in range(num_frames):
                masks_per_img = scaled_mask_preds[img_id][frame_id][
                    mask_indices]
                single_result = self.mask_head[-1].get_seg_masks_tracking(
                    masks_per_img, labels_per_img, scores_per_img,
                    torch.arange(self.test_cfg['max_per_img']), self.test_cfg,
                    img_metas[img_id])
                result.append(single_result)
            results.append(result)
        features = {
            'obj_feats': object_feats,
            'x_feats': x,
            'cls_scores': cls_score,
            'masks': mask_preds,
        }
        return results, features

    def init_weights(self):
        if self.init_cfg is not None and self.init_cfg[
                'type'] == 'Pretrained' and self.init_cfg['prefix'] is not None:
            from mmcv.cnn import initialize
            self.logger.info('Customized loading the tracker.')
            initialize(self, self.init_cfg)
        else:
            super().init_weights()
