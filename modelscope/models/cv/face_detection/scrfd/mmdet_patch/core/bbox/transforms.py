"""
The implementation here is modified based on insightface, originally MIT license and publicly available at
https://github.com/deepinsight/insightface/tree/master/detection/scrfd/mmdet/core/bbox/transforms.py
"""
import numpy as np
import torch


def bbox2result(bboxes, labels, num_classes, kps=None, num_kps=5):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    bbox_len = 5 if kps is None else 5 + num_kps * 2  # if has kps, add num_kps*2 into bbox
    if bboxes.shape[0] == 0:
        return [
            np.zeros((0, bbox_len), dtype=np.float32)
            for i in range(num_classes)
        ]
    else:
        if isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        if kps is None:
            return [bboxes[labels == i, :] for i in range(num_classes)]
        else:  # with kps
            if isinstance(kps, torch.Tensor):
                kps = kps.detach().cpu().numpy()
                return [
                    np.hstack([bboxes[labels == i, :], kps[labels == i, :]])
                    for i in range(num_classes)
                ]


def distance2kps(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded kps.
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)


def kps2distance(points, kps, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        kps (Tensor): Shape (n, K), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """

    preds = []
    for i in range(0, kps.shape[1], 2):
        px = kps[:, i] - points[:, i % 2]
        py = kps[:, i + 1] - points[:, i % 2 + 1]
        if max_dis is not None:
            px = px.clamp(min=0, max=max_dis - eps)
            py = py.clamp(min=0, max=max_dis - eps)
        preds.append(px)
        preds.append(py)
    return torch.stack(preds, -1)
