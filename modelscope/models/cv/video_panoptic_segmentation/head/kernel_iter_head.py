# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS, build_head
from mmdet.models.roi_heads import BaseRoIHead

from .kernel_update_head import VideoKernelUpdateHead
from .kernel_updator import KernelUpdator


@HEADS.register_module()
class VideoKernelIterHead(BaseRoIHead):

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
                 merge_joint=False,
                 num_proposals=100,
                 num_thing_classes=80,
                 num_stuff_classes=53,
                 mask_assign_stride=4,
                 ignore_label=255,
                 thing_label_in_seg=0,
                 with_track=False,
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
        self.merge_joint = merge_joint
        self.assign_stages = assign_stages
        self.do_panoptic = do_panoptic
        self.num_thing_classes = num_thing_classes
        self.num_stuff_classes = num_stuff_classes
        self.mask_assign_stride = mask_assign_stride
        self.thing_label_in_seg = thing_label_in_seg
        self.num_proposals = num_proposals
        self.ignore_label = ignore_label
        self.with_track = with_track
        super(VideoKernelIterHead, self).__init__(
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

    def init_weights(self):
        for i in range(self.num_stages):
            self.mask_head[i].init_weights()

    def init_assigner_sampler(self):
        pass

    def forward_train(self,
                      x,
                      proposal_feats,
                      mask_preds,
                      cls_score,
                      img_metas,
                      gt_masks,
                      gt_labels,
                      gt_pids=None,
                      gt_bboxes_ignore=None,
                      imgs_whwh=None,
                      gt_bboxes=None,
                      gt_sem_seg=None,
                      gt_sem_cls=None):
        pass

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
                      img_metas,
                      previous_obj_feats=None,
                      previous_mask_preds=None,
                      previous_x_feats=None):
        mask_head = self.mask_head[stage]
        cls_score, mask_preds, object_feats, x_feats, object_feats_track = mask_head(
            x,
            object_feats,
            mask_preds,
            img_metas=img_metas,
            previous_obj_feats=previous_obj_feats,
            previous_mask_preds=previous_mask_preds,
            previous_x_feats=previous_x_feats)
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
            object_feats=object_feats,
            object_feats_track=object_feats_track,
            x_feats=x_feats,
        )

        return mask_results

    def simple_test(self, x, proposal_feats, mask_preds, cls_score, img_metas):

        # Decode initial proposals
        num_imgs = len(img_metas)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            mask_results = self._mask_forward(stage, x, object_feats,
                                              mask_preds, img_metas)
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

        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id],
                                                  object_feats[img_id])
                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg['max_per_img'], sorted=True)
                mask_indices = topk_indices // num_classes
                labels_per_img = topk_indices % num_classes
                masks_per_img = scaled_mask_preds[img_id][mask_indices]
                single_result = self.mask_head[-1].get_seg_masks(
                    masks_per_img, labels_per_img, scores_per_img,
                    self.test_cfg, img_metas[img_id])
                results.append(single_result)

        if self.with_track:
            return results, object_feats, cls_score, mask_preds, scaled_mask_preds
        else:
            return results

    def simple_test_with_previous(
        self,
        x,
        proposal_feats,
        mask_preds,
        cls_score,
        img_metas,
        previous_obj_feats=None,
        previous_mask_preds=None,
        previous_x_feats=None,
        is_first=False,
    ):

        # Decode initial proposals
        num_imgs = len(img_metas)

        object_feats = proposal_feats
        for stage in range(self.num_stages):
            # only link the last stage inputs
            previous_obj_feats_cur = previous_obj_feats if stage == self.num_stages - 1 else None
            previous_mask_preds_cur = previous_mask_preds if stage == self.num_stages - 1 else None
            previous_x_feats_cur = previous_x_feats if stage == self.num_stages - 1 else None

            mask_results = self._mask_forward(
                stage,
                x,
                object_feats,
                mask_preds,
                img_metas,
                previous_obj_feats=previous_obj_feats_cur,
                previous_mask_preds=previous_mask_preds_cur,
                previous_x_feats=previous_x_feats_cur)
            object_feats = mask_results['object_feats']
            cls_score = mask_results['cls_score']
            mask_preds = mask_results['mask_preds']
            scaled_mask_preds = mask_results['scaled_mask_preds']
            object_feats_track = mask_results['object_feats_track']

        num_classes = self.mask_head[-1].num_classes
        results = []

        if self.mask_head[-1].loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
        else:
            cls_score = cls_score.softmax(-1)[..., :-1]

        if is_first:
            object_feats_track = object_feats

        if self.do_panoptic:
            for img_id in range(num_imgs):
                single_result = self.get_panoptic(cls_score[img_id],
                                                  scaled_mask_preds[img_id],
                                                  self.test_cfg,
                                                  img_metas[img_id],
                                                  object_feats_track[img_id])
                results.append(single_result)
        else:
            for img_id in range(num_imgs):
                cls_score_per_img = cls_score[img_id]
                scores_per_img, topk_indices = cls_score_per_img.flatten(
                    0, 1).topk(
                        self.test_cfg['max_per_img'], sorted=True)
                mask_indices = topk_indices // num_classes
                labels_per_img = topk_indices % num_classes
                masks_per_img = scaled_mask_preds[img_id][mask_indices]
                single_result = self.mask_head[-1].get_seg_masks(
                    masks_per_img, labels_per_img, scores_per_img,
                    self.test_cfg, img_metas[img_id])
                results.append(single_result)

        if self.with_track:
            return results, object_feats, cls_score, mask_preds, scaled_mask_preds
        else:
            return results

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

    def get_panoptic(self,
                     cls_scores,
                     mask_preds,
                     test_cfg,
                     img_meta,
                     obj_feat=None):
        # resize mask predictions back
        thing_scores = cls_scores[:self.num_proposals][:, :self.
                                                       num_thing_classes]
        thing_mask_preds = mask_preds[:self.num_proposals]
        thing_scores, topk_indices = thing_scores.flatten(0, 1).topk(
            self.test_cfg['max_per_img'], sorted=True)
        mask_indices = topk_indices // self.num_thing_classes
        thing_labels = topk_indices % self.num_thing_classes
        masks_per_img = thing_mask_preds[mask_indices]
        thing_masks = self.mask_head[-1].rescale_masks(masks_per_img, img_meta)

        # thing obj_feat
        thing_obj_feat = obj_feat[:self.num_proposals]
        thing_obj_feat = thing_obj_feat[mask_indices]

        if not self.merge_joint:
            thing_masks = thing_masks > test_cfg['mask_thr']
        bbox_result, segm_result, thing_mask_preds = self.mask_head[
            -1].segm2result(thing_masks, thing_labels, thing_scores)

        stuff_scores = cls_scores[
            self.num_proposals:][:, self.num_thing_classes:].diag()
        stuff_scores, stuff_inds = torch.sort(stuff_scores, descending=True)
        stuff_masks = mask_preds[self.num_proposals:][stuff_inds]
        stuff_masks = self.mask_head[-1].rescale_masks(stuff_masks, img_meta)

        # stuff obj_feat
        stuff_obj_feat = obj_feat[self.num_proposals:][stuff_inds]

        if not self.merge_joint:
            stuff_masks = stuff_masks > test_cfg['mask_thr']

        if self.merge_joint:
            stuff_labels = stuff_inds + self.num_thing_classes
            panoptic_result, thing_obj_feat = self.merge_stuff_thing_stuff_joint(
                thing_masks, thing_labels, thing_scores, stuff_masks,
                stuff_labels, stuff_scores, test_cfg['merge_stuff_thing'],
                thing_obj_feat, stuff_obj_feat)
        else:
            stuff_labels = stuff_inds + 1
            panoptic_result, thing_obj_feat = self.merge_stuff_thing_thing_first(
                thing_masks, thing_labels, thing_scores, stuff_masks,
                stuff_labels, stuff_scores, test_cfg['merge_stuff_thing'],
                thing_obj_feat, stuff_obj_feat)

        return bbox_result, segm_result, thing_mask_preds, panoptic_result, thing_obj_feat

    def merge_stuff_thing_thing_first(self,
                                      thing_masks,
                                      thing_labels,
                                      thing_scores,
                                      stuff_masks,
                                      stuff_labels,
                                      stuff_scores,
                                      merge_cfg=None,
                                      thing_obj_feat=None,
                                      stuff_obj_feat=None):

        H, W = thing_masks.shape[-2:]
        panoptic_seg = thing_masks.new_zeros((H, W), dtype=torch.int32)
        thing_masks = thing_masks.to(
            dtype=torch.bool, device=panoptic_seg.device)
        stuff_masks = stuff_masks.to(
            dtype=torch.bool, device=panoptic_seg.device)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-thing_scores)
        thing_obj_feat = thing_obj_feat[sorted_inds]
        current_segment_id = 0
        segments_info = []
        instance_ids = []

        # Add instances one-by-one, check for overlaps with existing ones
        for inst_id in sorted_inds:
            score = thing_scores[inst_id].item()
            if score < merge_cfg['instance_score_thr']:
                break
            mask = thing_masks[inst_id]  # H,W
            mask_area = mask.sum().item()

            if mask_area == 0:
                continue

            intersect = (mask > 0) & (panoptic_seg > 0)
            intersect_area = intersect.sum().item()

            if intersect_area * 1.0 / mask_area > merge_cfg['iou_thr']:
                continue

            if intersect_area > 0:
                mask = mask & (panoptic_seg == 0)

            mask_area = mask.sum().item()
            if mask_area == 0:
                continue

            current_segment_id += 1
            panoptic_seg[mask.bool()] = current_segment_id
            segments_info.append({
                'id': current_segment_id,
                'isthing': True,
                'score': score,
                'category_id': thing_labels[inst_id].item(),
                'instance_id': inst_id.item(),
            })
            instance_ids.append(inst_id.item())

        # Add semantic results to remaining empty areas
        sorted_inds = torch.argsort(-stuff_scores)
        sorted_stuff_labels = stuff_labels[sorted_inds]
        # paste semantic masks following the order of scores
        processed_label = []
        for semantic_label in sorted_stuff_labels:
            semantic_label = semantic_label.item()
            if semantic_label in processed_label:
                continue
            processed_label.append(semantic_label)
            sem_inds = stuff_labels == semantic_label
            sem_masks = stuff_masks[sem_inds].sum(0).bool()
            mask = sem_masks & (panoptic_seg == 0)
            mask_area = mask.sum().item()
            if mask_area < merge_cfg['stuff_max_area']:
                continue

            current_segment_id += 1
            panoptic_seg[mask] = current_segment_id
            segments_info.append({
                'id': current_segment_id,
                'isthing': False,
                'category_id': semantic_label,
                'area': mask_area,
            })
        return (panoptic_seg.cpu().numpy(),
                segments_info), thing_obj_feat[instance_ids]

    def merge_stuff_thing_stuff_joint(self,
                                      thing_masks,
                                      thing_labels,
                                      thing_scores,
                                      stuff_masks,
                                      stuff_labels,
                                      stuff_scores,
                                      merge_cfg=None,
                                      thing_obj=None,
                                      stuff_obj=None):

        H, W = thing_masks.shape[-2:]
        panoptic_seg = thing_masks.new_zeros((H, W), dtype=torch.int32)

        total_masks = torch.cat([thing_masks, stuff_masks], dim=0)
        total_scores = torch.cat([thing_scores, stuff_scores], dim=0)
        total_labels = torch.cat([thing_labels, stuff_labels], dim=0)
        obj_fea = torch.cat([thing_obj, stuff_obj], dim=0)

        cur_prob_masks = total_scores.view(-1, 1, 1) * total_masks
        segments_info = []
        cur_mask_ids = cur_prob_masks.argmax(0)

        # sort instance outputs by scores
        sorted_inds = torch.argsort(-total_scores)
        current_segment_id = 0
        sort_obj_fea = obj_fea
        things_ids = []
        for k in sorted_inds:
            pred_class = total_labels[k].item()
            isthing = pred_class < self.num_thing_classes
            if isthing and total_scores[k] < merge_cfg['instance_score_thr']:
                continue

            mask = cur_mask_ids == k
            mask_area = mask.sum().item()
            original_area = (total_masks[k] >= 0.5).sum().item()

            if mask_area > 0 and original_area > 0:
                if mask_area / original_area < merge_cfg['overlap_thr']:
                    continue
                current_segment_id += 1

                panoptic_seg[mask] = current_segment_id

                if isthing:
                    segments_info.append({
                        'id': current_segment_id,
                        'isthing': isthing,
                        'score': total_scores[k].item(),
                        'category_id': pred_class,  # 0, num_thing - 1
                        'instance_id': k.item(),
                    })
                    things_ids.append(k.item())
                else:
                    segments_info.append({
                        'id':
                        current_segment_id,
                        'isthing':
                        isthing,
                        'category_id':
                        pred_class - self.num_thing_classes
                        + 1,  # 1, num_stuff
                        'area':
                        mask_area,
                    })

        return (panoptic_seg.cpu().numpy(),
                segments_info), sort_obj_fea[things_ids]
