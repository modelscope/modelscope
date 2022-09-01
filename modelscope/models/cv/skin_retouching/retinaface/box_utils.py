# Implementation in this file is modifed from source code avaiable via https://github.com/ternaus/retinaface
from typing import List, Tuple, Union

import numpy as np
import torch


def point_form(boxes: torch.Tensor) -> torch.Tensor:
    """Convert prior_boxes to (x_min, y_min, x_max, y_max) representation for comparison to point form \
       ground truth data.

    Args:
        boxes: center-size default boxes from priorbox layers.
    Return:
        boxes: Converted x_min, y_min, x_max, y_max form of boxes.
    """
    return torch.cat(
        (boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2),
        dim=1)


def center_size(boxes: torch.Tensor) -> torch.Tensor:
    """Convert prior_boxes to (cx, cy, w, h) representation for comparison to center-size form ground truth data.
    Args:
        boxes: point_form boxes
    Return:
        boxes: Converted x_min, y_min, x_max, y_max form of boxes.
    """
    return torch.cat(
        ((boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]),
        dim=1)


def intersect(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """ We resize both tensors to [A,B,2] without new malloc:
    [A, 2] -> [A, 1, 2] -> [A, B, 2]
    [B, 2] -> [1, B, 2] -> [A, B, 2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: bounding boxes, Shape: [A, 4].
      box_b: bounding boxes, Shape: [B, 4].
    Return:
      intersection area, Shape: [A, B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    """Compute the jaccard overlap of two sets of boxes. The jaccard overlap is simply the intersection over
    union of two boxes.  Here we operate on ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_a = area_a.unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
    area_b = area_b.unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union


def matrix_iof(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    return iof of a and b, numpy version for data augmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    return area_i / np.maximum(area_a[:, np.newaxis], 1)


def match(
    threshold: float,
    box_gt: torch.Tensor,
    priors: torch.Tensor,
    variances: List[float],
    labels_gt: torch.Tensor,
    landmarks_gt: torch.Tensor,
    box_t: torch.Tensor,
    label_t: torch.Tensor,
    landmarks_t: torch.Tensor,
    batch_id: int,
) -> None:
    """Match each prior box with the ground truth box of the highest jaccard overlap, encode the bounding
    boxes, then return the matched indices corresponding to both confidence and location preds.

    Args:
        threshold: The overlap threshold used when matching boxes.
        box_gt: Ground truth boxes, Shape: [num_obj, 4].
        priors: Prior boxes from priorbox layers, Shape: [n_priors, 4].
        variances: Variances corresponding to each prior coord, Shape: [num_priors, 4].
        labels_gt: All the class labels for the image, Shape: [num_obj, 2].
        landmarks_gt: Ground truth landms, Shape [num_obj, 10].
        box_t: Tensor to be filled w/ endcoded location targets.
        label_t: Tensor to be filled w/ matched indices for labels predictions.
        landmarks_t: Tensor to be filled w/ endcoded landmarks targets.
        batch_id: current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landmarks preds.
    """
    # Compute iou between gt and priors
    overlaps = jaccard(box_gt, point_form(priors))
    # (Bipartite Matching)
    # [1, num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2
    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    if best_prior_idx_filter.shape[0] <= 0:
        box_t[batch_id] = 0
        label_t[batch_id] = 0
        return

    # [1, num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter,
                                   2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    matches = box_gt[best_truth_idx]  # Shape: [num_priors, 4]
    labels = labels_gt[best_truth_idx]  # Shape: [num_priors]
    # label as background
    labels[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)

    matches_landm = landmarks_gt[best_truth_idx]
    landmarks_gt = encode_landm(matches_landm, priors, variances)
    box_t[batch_id] = loc  # [num_priors, 4] encoded offsets to learn
    label_t[batch_id] = labels  # [num_priors] top class label for each prior
    landmarks_t[batch_id] = landmarks_gt


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= variances[0] * priors[:, 2:]
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def encode_landm(
        matched: torch.Tensor, priors: torch.Tensor,
        variances: Union[List[float], Tuple[float, float]]) -> torch.Tensor:
    """Encode the variances from the priorbox layers into the ground truth boxes we have matched
    (based on jaccard overlap) with the prior boxes.
    Args:
        matched: Coords of ground truth for each prior in point-form
            Shape: [num_priors, 10].
        priors: Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: Variances of priorboxes
    Return:
        encoded landmarks, Shape: [num_priors, 10]
    """

    # dist b/t match center and prior's center
    matched = torch.reshape(matched, (matched.size(0), 5, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0),
                                                 5).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0),
                                                 5).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0),
                                                5).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0),
                                                5).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    # encode variance
    g_cxcy = g_cxcy // variances[0] * priors[:, :, 2:]
    # return target for smooth_l1_loss
    return g_cxcy.reshape(g_cxcy.size(0), -1)


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc: torch.Tensor, priors: torch.Tensor,
           variances: Union[List[float], Tuple[float, float]]) -> torch.Tensor:
    """Decode locations from predictions using priors to undo the encoding we did for offset regression at train time.
    Args:
        loc: location predictions for loc layers,
            Shape: [num_priors, 4]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1]),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(
        pre: torch.Tensor, priors: torch.Tensor,
        variances: Union[List[float], Tuple[float, float]]) -> torch.Tensor:
    """Decode landmarks from predictions using priors to undo the encoding we did for offset regression at train time.
    Args:
        pre: landmark predictions for loc layers,
            Shape: [num_priors, 10]
        priors: Prior boxes in center-offset form.
            Shape: [num_priors, 4].
        variances: Variances of priorboxes
    Return:
        decoded landmark predictions
    """
    return torch.cat(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        dim=1,
    )


def log_sum_exp(x: torch.Tensor) -> torch.Tensor:
    """Utility function for computing log_sum_exp while determining This will be used to determine unaveraged
    confidence loss across all examples in a batch.
    Args:
        x: conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max
