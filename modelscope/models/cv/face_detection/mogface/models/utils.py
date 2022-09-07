# Modified from https://github.com/biubug6/Pytorch_Retinaface

import math
from itertools import product as product
from math import ceil

import numpy as np
import torch


def transform_anchor(anchors):
    """
    from [x0, x1, y0, y1] to [c_x, cy, w, h]
    x1 = x0 + w - 1
    c_x = (x0 + x1) / 2 = (2x0 + w - 1) / 2 = x0 + (w - 1) / 2
    """
    return np.concatenate(((anchors[:, :2] + anchors[:, 2:]) / 2,
                           anchors[:, 2:] - anchors[:, :2] + 1),
                          axis=1)


def normalize_anchor(anchors):
    """
    from  [c_x, cy, w, h] to [x0, x1, y0, y1]
    """
    item_1 = anchors[:, :2] - (anchors[:, 2:] - 1) / 2
    item_2 = anchors[:, :2] + (anchors[:, 2:] - 1) / 2
    return np.concatenate((item_1, item_2), axis=1)


class MogPriorBox(object):
    """
    both for fpn and single layer, single layer need to test
    return (np.array) [num_anchros, 4] [x0, y0, x1, y1]
    """

    def __init__(self,
                 scale_list=[1.],
                 aspect_ratio_list=[1.0],
                 stride_list=[4, 8, 16, 32, 64, 128],
                 anchor_size_list=[16, 32, 64, 128, 256, 512]):
        self.scale_list = scale_list
        self.aspect_ratio_list = aspect_ratio_list
        self.stride_list = stride_list
        self.anchor_size_list = anchor_size_list

    def __call__(self, img_height, img_width):
        final_anchor_list = []

        for idx, stride in enumerate(self.stride_list):
            anchor_list = []
            cur_img_height = img_height
            cur_img_width = img_width
            tmp_stride = stride

            while tmp_stride != 1:
                tmp_stride = tmp_stride // 2
                cur_img_height = (cur_img_height + 1) // 2
                cur_img_width = (cur_img_width + 1) // 2

            for i in range(cur_img_height):
                for j in range(cur_img_width):
                    for scale in self.scale_list:
                        cx = (j + 0.5) * stride
                        cy = (i + 0.5) * stride
                        side_x = self.anchor_size_list[idx] * scale
                        side_y = self.anchor_size_list[idx] * scale
                        for ratio in self.aspect_ratio_list:
                            anchor_list.append([
                                cx, cy, side_x / math.sqrt(ratio),
                                side_y * math.sqrt(ratio)
                            ])

            final_anchor_list.append(anchor_list)
        final_anchor_arr = np.concatenate(final_anchor_list, axis=0)
        normalized_anchor_arr = normalize_anchor(final_anchor_arr).astype(
            'float32')
        transformed_anchor = transform_anchor(normalized_anchor_arr)

        return transformed_anchor


class PriorBox(object):

    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.feature_maps = [[
            ceil(self.image_size[0] / step),
            ceil(self.image_size[1] / step)
        ] for step in self.steps]
        self.name = 's'

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [
                        x * self.steps[k] / self.image_size[1]
                        for x in [j + 0.5]
                    ]
                    dense_cy = [
                        y * self.steps[k] / self.image_size[0]
                        for y in [i + 0.5]
                    ]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]

        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def mogdecode(loc, anchors):
    """
    loc: torch.Tensor
    anchors: 2-d, torch.Tensor (cx, cy, w, h)
    boxes: 2-d, torch.Tensor (x0, y0, x1, y1)
    """

    boxes = torch.cat((anchors[:, :2] + loc[:, :2] * anchors[:, 2:],
                       anchors[:, 2:] * torch.exp(loc[:, 2:])), 1)

    boxes[:, 0] -= (boxes[:, 2] - 1) / 2
    boxes[:, 1] -= (boxes[:, 3] - 1) / 2
    boxes[:, 2] += boxes[:, 0] - 1
    boxes[:, 3] += boxes[:, 1] - 1

    return boxes


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat(
        (priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
         priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    a = priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:]
    b = priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:]
    c = priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:]
    d = priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:]
    e = priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:]
    landms = torch.cat((a, b, c, d, e), dim=1)
    return landms
