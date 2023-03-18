# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import build_assigner, build_sampler
from mmdet.datasets.coco_panoptic import INSTANCE_OFFSET
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead


@HEADS.register_module()
class KernelIterHeadVideo(BaseRoIHead):

    def __init__(self,
                 num_stages=6,
                 recursive=False,
                 assign_stages=5,
                 stage_loss_weights=(1, 1, 1, 1, 1, 1),
                 proposal_feature_channel=256,
                 merge_cls_scores=False,
                 do_panoptic=False,
                 post_assign=False,
                 hard_target=False,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 thing_label_in_seg=0,
                 mask_head=dict(
                     type='KernelUpdateHead',
                     num_classes=80,
                     num_fcs=2,
                     num_heads=8,
                     num_cls_fcs=1,
                     num_reg_fcs=3,
                     feedforward_channels=2048,
                     hidden_channels=256,
                     dropout=0.0,
                     roi_feat_size=7,
                     ffn_act_cfg=dict(type='ReLU', inplace=True)),
                 mask_out_stride=4,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        assert mask_head is not None
        assert len(stage_loss_weights) == num_stages
        self.num_stages = num_stages
        self.stage_loss_weights = stage_loss_weights
        self.proposal_feature_channel = proposal_feature_channel
        self.merge_cls_scores = merge_cls_scores
        self.recursive = recursive
        self.post_assign = post_assign
        self.mask_out_stride = mask_out_stride
        self.hard_target = hard_target
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_thing_classes + num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.num_proposals = num_proposals
        super().__init__(
            mask_head=mask_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            **kwargs)

    def init_bbox_head(self, mask_roi_extractor, mask_head):
        """Initialize box head and box roi extractor.

        Args:
            mask_roi_extractor (dict): Config of box roi extractor.
            mask_head (dict): Config of box in box head.
        """
        pass

    def init_assigner_sampler(self):
        """Initialize assigner and sampler for each stage."""
        self.mask_assigner = []
        self.mask_sampler = []
        if self.train_cfg is not None:
            for idx, rcnn_train_cfg in enumerate(self.train_cfg):
                self.mask_assigner.append(
                    build_assigner(rcnn_train_cfg.assigner))
                self.current_stage = idx
                self.mask_sampler.append(
                    build_sampler(rcnn_train_cfg.sampler, context=self))

    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def init_mask_head(self, mask_roi_extractor, mask_head):
        """Initialize mask head and mask roi extractor.

        Args:
            mask_roi_extractor (dict): Config of mask roi extractor.
            mask_head (dict): Config of mask in mask head.
        """
        self.mask_head = nn.ModuleList()
        if not isinstance(mask_head, list):
            mask_head = [mask_head for _ in range(self.num_stages)]
        assert len(mask_head) == self.num_stages
        for head in mask_head:
            self.mask_head.append(build_head(head))
        if self.recursive:
            for i in range(self.num_stages):
                self.mask_head[i] = self.mask_head[0]

    def _mask_forward(self,
                      stage,
                      x,
                      object_feats,
                      mask_preds,
                      img_metas=None):
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats = mask_head(
            x, object_feats, mask_preds, img_metas=img_metas)
        if mask_head.mask_upsample_stride > 1 and (stage == self.num_stages - 1
                                                   or self.training):
            scaled_mask_preds = F.interpolate(
                mask_preds,
                scale_factor=mask_head.mask_upsample_stride,
                align_corners=False,
                mode='bilinear')
        else:
            scaled_mask_preds = mask_preds

        mask_results = dict(
            cls_score=cls_score,
            mask_preds=mask_preds,
            scaled_mask_preds=scaled_mask_preds,
            object_feats=object_feats)
        return mask_results

    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      ref_img_metas,
                      gt_masks,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_bboxes=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None):

        num_imgs = len(ref_img_metas)
        num_frames = len(ref_img_metas[0])
        if self.mask_head[0].mask_upsample_stride > 1:
            prev_mask_preds = F.interpolate(
                mask_preds.detach(),
                scale_factor=self.mask_head[0].mask_upsample_stride,
                mode='bilinear',
                align_corners=False)
        else:
            prev_mask_preds = mask_preds.detach()

        if cls_score is not None:
            prev_cls_score = cls_score.detach()
        else:
            prev_cls_score = None

        if self.hard_target:
            gt_masks = [x.bool().float() for x in gt_masks]
        else:
            gt_masks = gt_masks

        object_feats = proposal_feats
        all_stage_loss = {}
        all_stage_mask_results = []
        assign_results = []
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(
                stage, x, object_feats, mask_preds, img_metas=None)
            all_stage_mask_results.append(mask_results)
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            cls_score = mask_results['cls_score']
            object_feats = mask_results['object_feats']

            if self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

            sampling_results = []
            if stage < self.assign_stages:
                assign_results = []
            for i in range(num_imgs):
                for j in range(num_frames):
                    if stage < self.assign_stages:
                        mask_for_assign = prev_mask_preds[
                            i * num_frames + j][:self.num_proposals]
                        if prev_cls_score is not None:
                            cls_for_assign = prev_cls_score[
                                i * num_frames + j][:self.num_proposals, :self.
                                                    num_thing_classes]
                        else:
                            cls_for_assign = None
                        assign_result = self.mask_assigner[stage].assign(
                            mask_for_assign,
                            cls_for_assign,
                            gt_masks[i][j],
                            gt_labels[i][:, 1][gt_labels[i][:, 0] == j],
                            img_meta=None)
                        assign_results.append(assign_result)
                    sampling_result = self.mask_sampler[stage].sample(
                        assign_results[i * num_frames + j],
                        scaled_mask_preds[i * num_frames + j], gt_masks[i][j])
                    sampling_results.append(sampling_result)
            mask_targets = self.mask_head[stage].get_targets(
                sampling_results,
                self.train_cfg[stage],
                True,
                gt_sem_seg=gt_sem_seg,
                gt_sem_cls=gt_sem_cls)

            single_stage_loss = self.mask_head[stage].loss(
                object_feats,
                cls_score,
                scaled_mask_preds,
                *mask_targets,
                imgs_whwh=imgs_whwh)
            for key, value in single_stage_loss.items():
                all_stage_loss[
                    f's{stage}_{key}'] = value * self.stage_loss_weights[stage]

            if not self.post_assign:
                prev_mask_preds = scaled_mask_preds.detach()
                prev_cls_score = cls_score.detach()

        bs_nf, num_query, c, ks1, ks2 = object_feats.size()
        bs_nf2, c2, h, w = x.size()
        assert ks1 == ks2
        assert bs_nf == bs_nf2
        assert bs_nf == num_frames * num_imgs
        assert c == c2
        features = {
            'obj_feats':
            object_feats.reshape(
                (num_imgs, num_frames, num_query, c, ks1, ks2)),
            # "x_feats":self.mask_head[-1].feat_transform(x).reshape((num_imgs, num_frames, c, h, w)),
            'x_feats':
            x.reshape((num_imgs, num_frames, c, h, w)),
            'cls_scores':
            cls_score.reshape(
                (num_imgs, num_frames, num_query, self.num_classes)),
            'masks':
            mask_preds.reshape((num_imgs, num_frames, num_query, h, w)),
        }
        return all_stage_loss, features

    def simple_test(self,
                    x,
                    proposal_feats,
                    mask_preds,
                    cls_score,
                    img_metas,
                    ref_img_metas,
                    imgs_whwh=None,
                    rescale=False):

        # Decode initial proposals
        num_imgs = len(ref_img_metas)
        num_frames = len(ref_img_metas[0])
        # num_proposals = proposal_feats.size(1)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        bs_nf, num_query, c, ks1, ks2 = object_feats.size()
        bs_nf2, c2, h, w = x.size()
        assert ks1 == ks2
        assert bs_nf == bs_nf2
        assert bs_nf == num_frames * num_imgs
        assert c == c2
        features = {
            'obj_feats':
            object_feats.reshape(
                (num_imgs, num_frames, num_query, c, ks1, ks2)),
            # "x_feats":self.mask_head[-1].feat_transform(x).reshape((num_imgs, num_frames, c, h, w)),
            'x_feats':
            x.reshape((num_imgs, num_frames, c, h, w)),
            'cls_scores':
            cls_score.reshape(
                (num_imgs, num_frames, num_query, self.num_classes)),
            'masks':
            mask_preds.reshape((num_imgs, num_frames, num_query, h, w)),
        }

        if self.do_panoptic:
            raise NotImplementedError
            # for img_id in range(num_imgs):
            #     single_result = self.get_panoptic(cls_score[img_id],
            #                                       scaled_mask_preds[img_id],
            #                                       self.test_cfg,
            #                                       ref_img_metas[img_id])
            #     results.append(single_result)
        else:
            for img_id in range(num_imgs):
                for frame_id in range(num_frames):
                    cls_score_per_img = cls_score[img_id * num_frames
                                                  + frame_id]
                    # h, quite tricky here, a bounding box can predict multiple results with different labels
                    scores_per_img, topk_indices = cls_score_per_img.flatten(
                        0, 1).topk(
                            self.test_cfg['max_per_img'], sorted=True)
                    mask_indices = topk_indices // num_classes
                    # Use the following when torch >= 1.9.0
                    # mask_indices = torch.div(topk_indices, num_classes, rounding_mode='floor')
                    labels_per_img = topk_indices % num_classes
                    masks_per_img = scaled_mask_preds[img_id * num_frames
                                                      + frame_id][mask_indices]
                    single_result = self.mask_head[-1].get_seg_masks(
                        masks_per_img, labels_per_img, scores_per_img,
                        self.test_cfg, img_metas[img_id])
                    results.append(single_result)
        return results, features

    def aug_test(self, features, proposal_list, img_metas, rescale=False):
        raise NotImplementedError('SparseMask does not support `aug_test`')

    def forward_dummy(self, x, proposal_boxes, proposal_feats, img_metas):
        """Dummy forward function when do the flops computing."""
        all_stage_mask_results = []
        num_imgs = len(img_metas)
        num_proposals = proposal_feats.size(1)
        C, H, W = x.shape[-3:]
        mask_preds = proposal_feats.bmm(x.view(num_imgs, C, -1)).view(
            num_imgs, num_proposals, H, W)
        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
            all_stage_mask_results.append(mask_results)
        return all_stage_mask_results

    def get_panoptic(self, cls_scores, mask_preds, test_cfg, img_meta):
        # resize mask predictions back
        scores = cls_scores[:self.num_proposals][:, :self.num_thing_classes]
        thing_scores, thing_labels = scores.max(dim=1)
        stuff_scores = cls_scores[
            self.num_proposals:][:, self.num_thing_classes:].diag()
        stuff_labels = torch.arange(
            0, self.num_stuff_classes) + self.num_thing_classes
        stuff_labels = stuff_labels.to(thing_labels.device)

        total_masks = self.mask_head[-1].rescale_masks(mask_preds, img_meta)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)

        panoptic_result = self.merge_stuff_thing(total_masks, total_labels,
                                                 total_scores,
                                                 test_cfg.merge_stuff_thing)
        return dict(pan_results=panoptic_result)

    def merge_stuff_thing(self,
                          total_masks,
                          total_labels,
                          total_scores,
                          merge_cfg=None):

        H, W = total_masks.shape[-2:]
        panoptic_seg = total_masks.new_full((H, W),
                                            self.num_classes,
                                            dtype=torch.long)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0

        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg.instance_score_thr:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg.overlap_thr:
                    continue

                panoptic_seg[mask] = total_labels[k] \
                    + current_segment_id * INSTANCE_OFFSET
                current_segment_id += 1

        return panoptic_seg.cpu().numpy()
