# ------------------------------------------------------------------------------
# Part of implementation is adopted from CenterNet,
# made publicly available under the MIT License at https://github.com/xingyizhou/CenterNet.git
# ------------------------------------------------------------------------------

import cv2
import numpy as np
import shapely
import torch
import torch.nn as nn
from shapely.geometry import MultiPoint, Point, Polygon


def _gather_feat(feat, ind, mask=None):
    # mandatory
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    # mandatory
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def _get_4ps_feat(cc_match, output):
    # mandatory
    if isinstance(output, dict):
        feat = output['cr']
    else:
        feat = output
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.contiguous().view(feat.size(0), -1, feat.size(3))
    feat = feat.unsqueeze(3).expand(
        feat.size(0), feat.size(1), feat.size(2), 4)

    dim = feat.size(2)

    cc_match = cc_match.unsqueeze(2).expand(
        cc_match.size(0), cc_match.size(1), dim, cc_match.size(2))
    if not (isinstance(output, dict)):
        cc_match = torch.where(
            cc_match < feat.shape[1], cc_match, (feat.shape[0] - 1)
            * torch.ones(cc_match.shape).to(torch.int64).cuda())
        cc_match = torch.where(
            cc_match >= 0, cc_match,
            torch.zeros(cc_match.shape).to(torch.int64).cuda())
    feat = feat.gather(1, cc_match)
    return feat


def _nms(heat, name, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    # save_map(hmax.cpu().numpy()[0],name)
    keep = (hmax == heat).float()
    return heat * keep, keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (
        torch.Tensor([height]).to(torch.int64).cuda()
        * torch.Tensor([width]).to(torch.int64).cuda())
    topk_ys = (topk_inds / torch.Tensor([width]).cuda()).int().float()
    topk_xs = (topk_inds
               % torch.Tensor([width]).to(torch.int64).cuda()).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind // K).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1),
                             topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def corner_decode(mk, st_reg, mk_reg=None, K=400):
    batch, cat, height, width = mk.size()
    mk, keep = _nms(mk, 'mk.0.maxpool')
    scores, inds, clses, ys, xs = _topk(mk, K=K)
    if mk_reg is not None:
        reg = _tranpose_and_gather_feat(mk_reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5
    scores = scores.view(batch, K, 1)
    st_Reg = _tranpose_and_gather_feat(st_reg, inds)
    bboxes_vec = [
        xs - st_Reg[..., 0:1], ys - st_Reg[..., 1:2], xs - st_Reg[..., 2:3],
        ys - st_Reg[..., 3:4], xs - st_Reg[..., 4:5], ys - st_Reg[..., 5:6],
        xs - st_Reg[..., 6:7], ys - st_Reg[..., 7:8]
    ]
    bboxes = torch.cat(bboxes_vec, dim=2)
    corner_dict = {
        'scores': scores,
        'inds': inds,
        'ys': ys,
        'xs': xs,
        'gboxes': bboxes
    }
    return scores, inds, ys, xs, bboxes, corner_dict


def ctdet_4ps_decode(heat,
                     wh,
                     ax,
                     cr,
                     corner_dict=None,
                     reg=None,
                     cat_spec_wh=False,
                     K=100,
                     wiz_rev=False):

    batch, cat, height, width = heat.size()
    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat, keep = _nms(heat, 'hm.0.maxpool')

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
    ax = _tranpose_and_gather_feat(ax, inds)

    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 8)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 8).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 8)
    else:
        wh = wh.view(batch, K, 8)
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes_vec = [
        xs - wh[..., 0:1], ys - wh[..., 1:2], xs - wh[..., 2:3],
        ys - wh[..., 3:4], xs - wh[..., 4:5], ys - wh[..., 5:6],
        xs - wh[..., 6:7], ys - wh[..., 7:8]
    ]
    bboxes = torch.cat(bboxes_vec, dim=2)

    cc_match = torch.cat(
        [(xs - wh[..., 0:1]) + width * torch.round(ys - wh[..., 1:2]),
         (xs - wh[..., 2:3]) + width * torch.round(ys - wh[..., 3:4]),
         (xs - wh[..., 4:5]) + width * torch.round(ys - wh[..., 5:6]),
         (xs - wh[..., 6:7]) + width * torch.round(ys - wh[..., 7:8])],
        dim=2)

    cc_match = torch.round(cc_match).to(torch.int64)

    cr_feat = _get_4ps_feat(cc_match, cr)
    cr_feat = cr_feat.sum(axis=3)

    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections, keep, ax, cr_feat


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift  # [0,0] #
    src[1, :] = center + src_dir + scale_tmp * shift  # scale #
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]  # [0,0] #
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5],
                         np.float32) + dst_dir  # output_size #

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_affine_transform_upper_left(center,
                                    scale,
                                    rot,
                                    output_size,
                                    shift=np.array([0, 0], dtype=np.float32),
                                    inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    dst[0, :] = [0, 0]
    if center[0] < center[1]:
        src[1, :] = [scale[0], center[1]]
        dst[1, :] = [output_size[0], 0]
    else:
        src[1, :] = [center[0], scale[0]]
        dst[1, :] = [0, output_size[0]]
    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def transform_preds_upper_left(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)

    trans = get_affine_transform_upper_left(
        center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def ctdet_4ps_post_process_upper_left(dets, c, s, h, w, num_classes, rot=0):
    # dets: batch x max_dets x dim
    # return 1-based class det dict
    ret = []
    for i in range(dets.shape[0]):
        top_preds = {}
        dets[i, :, 0:2] = transform_preds_upper_left(dets[i, :, 0:2], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 2:4] = transform_preds_upper_left(dets[i, :, 2:4], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 4:6] = transform_preds_upper_left(dets[i, :, 4:6], c[i],
                                                     s[i], (w, h), rot)
        dets[i, :, 6:8] = transform_preds_upper_left(dets[i, :, 6:8], c[i],
                                                     s[i], (w, h), rot)
        classes = dets[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            tmp_top_pred = [
                dets[i, inds, :8].astype(np.float32),
                dets[i, inds, 8:9].astype(np.float32)
            ]
            top_preds[j + 1] = np.concatenate(tmp_top_pred, axis=1).tolist()
        ret.append(top_preds)
    return ret


def ctdet_corner_post_process(corner_st_reg, c, s, h, w, num_classes):
    for i in range(corner_st_reg.shape[0]):
        corner_st_reg[i, :, 0:2] = transform_preds(corner_st_reg[i, :, 0:2],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 2:4] = transform_preds(corner_st_reg[i, :, 2:4],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 4:6] = transform_preds(corner_st_reg[i, :, 4:6],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 6:8] = transform_preds(corner_st_reg[i, :, 6:8],
                                                   c[i], s[i], (w, h))
        corner_st_reg[i, :, 8:10] = transform_preds(corner_st_reg[i, :, 8:10],
                                                    c[i], s[i], (w, h))
    return corner_st_reg


def merge_outputs(detections):
    # thresh_conf, thresh_min, thresh_max = 0.1, 0.5, 0.7
    num_classes, max_per_image = 2, 3000
    results = {}
    for j in range(1, num_classes + 1):
        results[j] = np.concatenate([detection[j] for detection in detections],
                                    axis=0).astype(np.float32)
    scores = np.hstack([results[j][:, 8] for j in range(1, num_classes + 1)])
    if len(scores) > max_per_image:
        kth = len(scores) - max_per_image
        thresh = np.partition(scores, kth)[kth]
        for j in range(1, num_classes + 1):
            keep_inds = (results[j][:, 8] >= thresh)
            results[j] = results[j][keep_inds]
    return results


def filter(results, logi, ps):
    # this function select boxes
    batch_size, feat_dim = logi.shape[0], logi.shape[2]

    num_valid = sum(results[1][:, 8] >= 0.15)

    slct_logi = np.zeros((batch_size, num_valid, feat_dim), dtype=np.float32)
    slct_dets = np.zeros((batch_size, num_valid, 8), dtype=np.int32)
    for i in range(batch_size):
        for j in range(num_valid):
            slct_logi[i, j, :] = logi[i, j, :].cpu()
            slct_dets[i, j, :] = ps[i, j, :].cpu()

    return torch.Tensor(slct_logi).cuda(), torch.Tensor(slct_dets).cuda()


def normalized_ps(ps, vocab_size):
    ps = torch.round(ps).to(torch.int64)
    ps = torch.where(ps < vocab_size, ps, (vocab_size - 1)
                     * torch.ones(ps.shape).to(torch.int64).cuda())
    ps = torch.where(ps >= 0, ps, torch.zeros(ps.shape).to(torch.int64).cuda())
    return ps


def process_detect_output(output, meta):
    K, MK = 3000, 5000
    num_classes = 2
    scale = 1.0

    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']
    st = output['st']
    ax = output['ax']
    cr = output['cr']

    scores, inds, ys, xs, st_reg, corner_dict = corner_decode(
        hm[:, 1:2, :, :], st, reg, K=MK)
    dets, keep, logi, cr = ctdet_4ps_decode(
        hm[:, 0:1, :, :], wh, ax, cr, corner_dict, reg=reg, K=K, wiz_rev=False)
    raw_dets = dets
    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1, -1, dets.shape[2])
    dets = ctdet_4ps_post_process_upper_left(dets.copy(),
                                             [meta['c'].cpu().numpy()],
                                             [meta['s']], meta['out_height'],
                                             meta['out_width'], 2)
    for j in range(1, num_classes + 1):
        dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 9)
        dets[0][j][:, :8] /= scale
    dets = dets[0]
    detections = [dets]

    logi = logi + cr
    results = merge_outputs(detections)
    slct_logi_feat, slct_dets_feat = filter(results, logi, raw_dets[:, :, :8])
    slct_dets_feat = normalized_ps(slct_dets_feat, 256)
    slct_output_dets = results[1][:slct_logi_feat.shape[1], :8]

    return slct_logi_feat, slct_dets_feat, slct_output_dets


def process_logic_output(logi):
    logi_floor = logi.floor()
    dev = logi - logi_floor
    logi = torch.where(dev > 0.5, logi_floor + 1, logi_floor)

    return logi


def load_lore_model(model, checkpoint, mtype):
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            if mtype == 'model':
                if k.startswith('model'):
                    state_dict[k[6:]] = state_dict_[k]
                else:
                    continue
            else:
                if k.startswith('processor'):
                    state_dict[k[10:]] = state_dict_[k]
                else:
                    continue
    model_state_dict = model.state_dict()
    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(k, model_state_dict[k].shape,
                                               state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)
