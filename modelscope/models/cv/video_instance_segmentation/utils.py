# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net follow the MIT license
import numpy as np
import torch
from mmdet.core import bbox2result


def sem2ins_masks(gt_sem_seg, num_thing_classes=80):
    """Convert semantic segmentation mask to binary masks

    Args:
        gt_sem_seg (torch.Tensor): Semantic masks to be converted.
            [0, num_thing_classes-1] is the classes of things,
            [num_thing_classes:] is the classes of stuff.
        num_thing_classes (int, optional): Number of thing classes.
            Defaults to 80.

    Returns:
        tuple[torch.Tensor]: (mask_labels, bin_masks).
            Mask labels and binary masks of stuff classes.
    """
    # gt_sem_seg is zero-started, where zero indicates the first class
    # since mmdet>=2.17.0, see more discussion in
    # https://mmdetection.readthedocs.io/en/latest/conventions.html#coco-panoptic-dataset  # noqa
    classes = torch.unique(gt_sem_seg)
    # classes ranges from 0 - N-1, where the class IDs in
    # [0, num_thing_classes - 1] are IDs of thing classes
    masks = []
    labels = []

    for i in classes:
        # skip ignore class 255 and "thing classes" in semantic seg
        if i == 255 or i < num_thing_classes:
            continue
        labels.append(i)
        masks.append(gt_sem_seg == i)

    if len(labels) > 0:
        labels = torch.stack(labels)
        masks = torch.cat(masks)
    else:
        labels = gt_sem_seg.new_zeros(size=[0])
        masks = gt_sem_seg.new_zeros(
            size=[0, gt_sem_seg.shape[-2], gt_sem_seg.shape[-1]])
    return labels.long(), masks.float()


def outs2results(bboxes=None,
                 labels=None,
                 masks=None,
                 ids=None,
                 num_classes=None,
                 **kwargs):
    """Convert tracking/detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        masks (torch.Tensor | np.ndarray): shape (n, h, w)
        ids (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, not including background class
    Returns:
        dict[str : list(ndarray) | list[list[np.ndarray]]]: tracking/detection
        results of each class. It may contain keys as belows:
        - bbox_results (list[np.ndarray]): Each list denotes bboxes of one
            category.
        - mask_results (list[list[np.ndarray]]): Each outer list denotes masks
            of one category. Each inner list denotes one mask belonging to
            the category. Each mask has shape (h, w).
    """
    assert labels is not None
    assert num_classes is not None

    results = dict()

    if ids is not None:
        valid_inds = ids > -1
        ids = ids[valid_inds]
        labels = labels[valid_inds]

    if bboxes is not None:
        if ids is not None:
            bboxes = bboxes[valid_inds]
            if bboxes.shape[0] == 0:
                bbox_results = [
                    np.zeros((0, 6), dtype=np.float32)
                    for i in range(num_classes)
                ]
            else:
                if isinstance(bboxes, torch.Tensor):
                    bboxes = bboxes.cpu().numpy()
                    labels = labels.cpu().numpy()
                    ids = ids.cpu().numpy()
                bbox_results = [
                    np.concatenate(
                        (ids[labels == i, None], bboxes[labels == i, :]),
                        axis=1) for i in range(num_classes)
                ]
        else:
            bbox_results = bbox2result(bboxes, labels, num_classes)
        results['bbox_results'] = bbox_results

    if masks is not None:
        if ids is not None:
            masks = masks[valid_inds]
        if isinstance(masks, torch.Tensor):
            masks = masks.detach().cpu().numpy()
        masks_results = [[] for _ in range(num_classes)]
        for i in range(bboxes.shape[0]):
            masks_results[labels[i]].append(masks[i])
        results['mask_results'] = masks_results

    return results
