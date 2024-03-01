# @Description: Loss Functions (Sec 3.4 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07
# @https://github.com/doublez0108/geomvsnet

import torch


def geomvsnet_loss(inputs, depth_gt_ms, mask_ms, **kwargs):

    stage_lw = kwargs.get('stage_lw', [1, 1, 1, 1])
    depth_values = kwargs.get('depth_values')
    depth_min, depth_max = depth_values[:, 0], depth_values[:, -1]

    total_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms['stage1'].device,
        requires_grad=False)
    pw_loss_stages = []
    dds_loss_stages = []
    for stage_idx, (stage_inputs, stage_key) in enumerate([
        (inputs[k], k) for k in inputs.keys() if 'stage' in k
    ]):

        depth = stage_inputs['depth_filtered']
        prob_volume = stage_inputs['prob_volume']
        depth_value = stage_inputs['depth_hypo']

        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key] > 0.5

        # pw loss
        pw_loss = pixel_wise_loss(prob_volume, depth_gt, mask, depth_value)
        pw_loss_stages.append(pw_loss)

        # dds loss
        dds_loss = depth_distribution_similarity_loss(depth, depth_gt, mask,
                                                      depth_min, depth_max)
        dds_loss_stages.append(dds_loss)

        # total loss
        lam1, lam2 = 0.8, 0.2
        total_loss = total_loss + stage_lw[stage_idx] * (
            lam1 * pw_loss + lam2 * dds_loss)

    depth_pred = stage_inputs['depth']
    depth_gt = depth_gt_ms[stage_key]
    epe = cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max)

    return total_loss, epe, pw_loss_stages, dds_loss_stages


def pixel_wise_loss(prob_volume, depth_gt, mask, depth_value):
    mask_true = mask
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-12

    shape = depth_gt.shape

    depth_num = depth_value.shape[1]
    depth_value_mat = depth_value

    gt_index_image = torch.argmin(
        torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)

    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1],
                                  shape[2]).type(mask_true.type()).scatter_(
                                      1, gt_index_image, 1)
    cross_entropy_image = -torch.sum(
        gt_index_volume * torch.log(prob_volume + 1e-12), dim=1).squeeze(1)
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)

    pw_loss = masked_cross_entropy
    return pw_loss


def depth_distribution_similarity_loss(depth, depth_gt, mask, depth_min,
                                       depth_max):
    depth_norm = depth * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    M_bins = 48
    kl_min = torch.min(torch.min(depth_gt), depth.mean() - 3. * depth.std())
    kl_max = torch.max(torch.max(depth_gt), depth.mean() + 3. * depth.std())
    bins = torch.linspace(kl_min, kl_max, steps=M_bins)

    kl_divs = []
    for i in range(len(bins) - 1):
        bin_mask = (depth_gt >= bins[i]) & (depth_gt < bins[i + 1])
        merged_mask = mask & bin_mask

        if merged_mask.sum() > 0:
            p = depth_norm[merged_mask]
            q = depth_gt_norm[merged_mask]
            kl_div = torch.nn.functional.kl_div(
                torch.log(p) - torch.log(q), p, reduction='batchmean')
            kl_div = torch.log(kl_div)
            kl_divs.append(kl_div)

    dds_loss = sum(kl_divs)
    return dds_loss


def cal_metrics(depth_pred, depth_gt, mask, depth_min, depth_max):
    depth_pred_norm = depth_pred * 128 / (depth_max - depth_min)[:, None, None]
    depth_gt_norm = depth_gt * 128 / (depth_max - depth_min)[:, None, None]

    abs_err = torch.abs(depth_pred_norm[mask] - depth_gt_norm[mask])
    epe = abs_err.mean()
    # err1 = (abs_err <= 1).float().mean() * 100
    # err3 = (abs_err <= 3).float().mean() * 100

    return epe  # err1, err3
