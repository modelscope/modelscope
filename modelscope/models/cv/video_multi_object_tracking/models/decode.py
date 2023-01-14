# The implementation is adopted from FairMOT,
# made publicly available under the MIT License at https://github.com/ifzhang/FairMOT
import torch
import torch.nn as nn

from modelscope.models.cv.video_multi_object_tracking.utils.utils import (
    _gather_feat, _tranpose_and_gather_feat)


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = torch.true_divide(topk_inds, width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = torch.true_divide(topk_ind, K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def mot_decode(heat, wh, reg=None, ltrb=False, K=100):
    batch, cat, height, width = heat.size()

    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    if ltrb:
        wh = wh.view(batch, K, 4)
    else:
        wh = wh.view(batch, K, 2)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if ltrb:
        a = xs - wh[..., 0:1]
        b = ys - wh[..., 1:2]
        c = xs + wh[..., 2:3]
        d = ys + wh[..., 3:4]
        bboxes = torch.cat([a, b, c, d], dim=2)
    else:
        a = xs - wh[..., 0:1] / 2
        b = ys - wh[..., 1:2] / 2
        c = xs + wh[..., 0:1] / 2
        d = ys + wh[..., 1:2] / 2
        bboxes = torch.cat([a, b, c, d], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, inds
