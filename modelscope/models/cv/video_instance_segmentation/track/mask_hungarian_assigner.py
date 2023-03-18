# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license

import numpy as np
import torch
from mmdet.core import AssignResult, BaseAssigner
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import MATCH_COST, build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@MATCH_COST.register_module()
class MaskCost(object):
    """MaskCost.

    Args:
        weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1., pred_act=False, act_mode='sigmoid'):
        self.weight = weight
        self.pred_act = pred_act
        self.act_mode = act_mode

    def __call__(self, cls_pred, target):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        if self.pred_act and self.act_mode == 'sigmoid':
            cls_pred = cls_pred.sigmoid()
        elif self.pred_act:
            cls_pred = cls_pred.softmax(dim=0)

        _, H, W = target.shape
        # flatten_cls_pred = cls_pred.view(num_proposals, -1)
        # eingum is ~10 times faster than matmul
        pos_cost = torch.einsum('nhw,mhw->nm', cls_pred, target)
        neg_cost = torch.einsum('nhw,mhw->nm', 1 - cls_pred, 1 - target)
        cls_cost = -(pos_cost + neg_cost) / (H * W)
        return cls_cost * self.weight


@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssignerVideo(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classfication cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_weight (int | float, optional): The scale factor for classification
            cost. Default 1.0.
        bbox_weight (int | float, optional): The scale factor for regression
            L1 cost. Default 1.0.
        iou_weight (int | float, optional): The scale factor for regression
            iou cost. Default 1.0.
        iou_calculator (dict | optional): The config for the iou calculation.
            Default type `BboxOverlaps2D`.
        iou_mode (str | optional): "iou" (intersection over union), "iof"
                (intersection over foreground), or "giou" (generalized
                intersection over union). Default "giou".
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 mask_cost=dict(type='SigmoidCost', weight=1.0),
                 dice_cost=dict(),
                 boundary_cost=None,
                 topk=1):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        if boundary_cost is not None:
            self.boundary_cost = build_match_cost(boundary_cost)
        else:
            self.boundary_cost = None
        self.topk = topk

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_instance_ids,
               img_meta=None,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        instances = torch.unique(gt_instance_ids[:, 1])
        num_frames = bbox_pred.size(0)
        h, w = bbox_pred.shape[-2:]
        gt_masks = []
        gt_labels_tensor = []
        for instance_id in instances:
            temp = gt_instance_ids[gt_instance_ids[:, 1] == instance_id, 0]
            gt_instance_frame_ids = temp
            instance_masks = []
            gt_label_id = None
            for frame_id in range(num_frames):
                gt_frame_instance_ids = gt_instance_ids[
                    gt_instance_ids[:, 0] == frame_id, 1]
                gt_frame_label_ids = gt_labels[gt_labels[:, 0] == frame_id, 1]
                assert len(gt_frame_label_ids) == len(gt_frame_label_ids)
                if not (frame_id in gt_instance_frame_ids):
                    gt_mask_frame = torch.zeros(
                        (h, w),
                        device=gt_instance_frame_ids.device,
                        dtype=torch.float)
                else:
                    gt_index = torch.nonzero(
                        (gt_frame_instance_ids == instance_id),
                        as_tuple=True)[0].item()
                    gt_mask_frame = gt_bboxes[frame_id][gt_index]
                    gt_label_id = gt_frame_label_ids[gt_index].item(
                    ) if gt_label_id is None else gt_label_id
                    assert gt_label_id == gt_frame_label_ids[gt_index].item()
                instance_masks.append(gt_mask_frame)
            gt_masks.append(torch.stack(instance_masks))
            gt_labels_tensor.append(gt_label_id)
        gt_masks = torch.stack(gt_masks)
        gt_labels_tensor = torch.tensor(
            gt_labels_tensor, device=gt_masks.device, dtype=torch.long)

        num_gts, num_bboxes = len(instances), bbox_pred.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        pred_masks_match = torch.einsum('fqhw->qfhw', bbox_pred).reshape(
            (num_bboxes, -1, w))
        gt_masks_match = gt_masks.reshape((num_gts, -1, w))
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels_tensor)
        else:
            cls_cost = 0
        if self.mask_cost.weight != 0:
            reg_cost = self.mask_cost(pred_masks_match, gt_masks_match)
        else:
            reg_cost = 0
        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(pred_masks_match, gt_masks_match)
        else:
            dice_cost = 0
        if self.boundary_cost is not None and self.boundary_cost.weight != 0:
            b_cost = self.boundary_cost(pred_masks_match, gt_masks_match)
        else:
            b_cost = 0
        cost = cls_cost + reg_cost + dice_cost + b_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        if self.topk == 1:
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        else:
            topk_matched_row_inds = []
            topk_matched_col_inds = []
            for i in range(self.topk):
                matched_row_inds, matched_col_inds = linear_sum_assignment(
                    cost)
                topk_matched_row_inds.append(matched_row_inds)
                topk_matched_col_inds.append(matched_col_inds)
                cost[matched_row_inds] = 1e10
            matched_row_inds = np.concatenate(topk_matched_row_inds)
            matched_col_inds = np.concatenate(topk_matched_col_inds)

        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels_tensor[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None,
            labels=assigned_labels), gt_masks_match
