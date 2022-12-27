# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib.cm import get_cmap

from modelscope.models.cv.video_depth_estimation.utils.image import (
    flip_lr, gradient_x, gradient_y, interpolate_image, load_image)
from modelscope.models.cv.video_depth_estimation.utils.types import (is_seq,
                                                                     is_tensor)


def load_depth(file):
    """
    Load a depth map from file
    Parameters
    ----------
    file : str
        Depth map filename (.npz or .png)

    Returns
    -------
    depth : np.array [H,W]
        Depth map (invalid pixels are 0)
    """
    if file.endswith('npz'):
        return np.load(file)['depth']
    elif file.endswith('png'):
        depth_png = np.array(load_image(file), dtype=int)
        assert (np.max(depth_png) > 255), 'Wrong .png depth file'
        return depth_png.astype(np.float) / 256.
    else:
        raise NotImplementedError('Depth extension not supported.')


def write_depth(filename, depth, intrinsics=None):
    """
    Write a depth map to file, and optionally its corresponding intrinsics.

    Parameters
    ----------
    filename : str
        File where depth map will be saved (.npz or .png)
    depth : np.array [H,W]
        Depth map
    intrinsics : np.array [3,3]
        Optional camera intrinsics matrix
    """
    # If depth is a tensor
    if is_tensor(depth):
        depth = depth.detach().squeeze().cpu()
    # If intrinsics is a tensor
    if is_tensor(intrinsics):
        intrinsics = intrinsics.detach().cpu()
    # If we are saving as a .npz
    if filename.endswith('.npz'):
        np.savez_compressed(filename, depth=depth, intrinsics=intrinsics)
    # If we are saving as a .png
    elif filename.endswith('.png'):
        depth = transforms.ToPILImage()((depth * 256).int())
        depth.save(filename)
    # Something is wrong
    else:
        raise NotImplementedError('Depth filename not valid.')


def viz_inv_depth(inv_depth,
                  normalizer=None,
                  percentile=95,
                  colormap='plasma',
                  filter_zeros=False):
    """
    Converts an inverse depth map to a colormap for visualization.

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization

    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth,
            percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]


def inv2depth(inv_depth):
    """
    Invert an inverse depth map to produce a depth map

    Parameters
    ----------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map

    Returns
    -------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map
    """
    if is_seq(inv_depth):
        return [inv2depth(item) for item in inv_depth]
    else:
        depth = 1. / inv_depth.clamp(min=1e-6)
        depth[inv_depth <= 0.] = 0.
        return depth


def depth2inv(depth):
    """
    Invert a depth map to produce an inverse depth map

    Parameters
    ----------
    depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Depth map

    Returns
    -------
    inv_depth : torch.Tensor or list of torch.Tensor [B,1,H,W]
        Inverse depth map

    """
    if is_seq(depth):
        return [depth2inv(item) for item in depth]
    else:
        inv_depth = 1. / depth.clamp(min=1e-6)
        inv_depth[depth <= 0.] = 0.
        return inv_depth


def inv_depths_normalize(inv_depths):
    """
    Inverse depth normalization

    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps

    Returns
    -------
    norm_inv_depths : list of torch.Tensor [B,1,H,W]
        Normalized inverse depth maps
    """
    mean_inv_depths = [
        inv_depth.mean(2, True).mean(3, True) for inv_depth in inv_depths
    ]
    return [
        inv_depth / mean_inv_depth.clamp(min=1e-6)
        for inv_depth, mean_inv_depth in zip(inv_depths, mean_inv_depths)
    ]


def calc_smoothness(inv_depths, images, num_scales):
    """
    Calculate smoothness values for inverse depths

    Parameters
    ----------
    inv_depths : list of torch.Tensor [B,1,H,W]
        Inverse depth maps
    images : list of torch.Tensor [B,3,H,W]
        Inverse depth maps
    num_scales : int
        Number of scales considered

    Returns
    -------
    smoothness_x : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction x
    smoothness_y : list of torch.Tensor [B,1,H,W]
        Smoothness values in direction y
    """
    inv_depths_norm = inv_depths_normalize(inv_depths)
    inv_depth_gradients_x = [gradient_x(d) for d in inv_depths_norm]
    inv_depth_gradients_y = [gradient_y(d) for d in inv_depths_norm]

    image_gradients_x = [gradient_x(image) for image in images]
    image_gradients_y = [gradient_y(image) for image in images]

    weights_x = [
        torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
        for g in image_gradients_x
    ]
    weights_y = [
        torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True))
        for g in image_gradients_y
    ]

    # Note: Fix gradient addition
    smoothness_x = [
        inv_depth_gradients_x[i] * weights_x[i] for i in range(num_scales)
    ]
    smoothness_y = [
        inv_depth_gradients_y[i] * weights_y[i] for i in range(num_scales)
    ]
    return smoothness_x, smoothness_y


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_inv_depth(inv_depth, inv_depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = inv_depth.shape
    inv_depth_hat = flip_lr(inv_depth_flipped)
    inv_depth_fused = fuse_inv_depth(inv_depth, inv_depth_hat, method=method)
    xs = torch.linspace(
        0., 1., W, device=inv_depth.device,
        dtype=inv_depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * inv_depth + mask * inv_depth_hat + \
        (1.0 - mask - mask_hat) * inv_depth_fused


def compute_depth_metrics(config, gt, pred, use_gt_scale=True):
    """
    Compute depth metrics from predicted and ground-truth depth maps

    Parameters
    ----------
    config : CfgNode
        Metrics parameters
    gt : torch.Tensor [B,1,H,W]
        Ground-truth depth map
    pred : torch.Tensor [B,1,H,W]
        Predicted depth map
    use_gt_scale : bool
        True if ground-truth median-scaling is to be used

    Returns
    -------
    metrics : torch.Tensor [7]
        Depth metrics (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    crop = config.crop == 'garg'

    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    SI = SILog = iabs_diff = irmse = abs_diff = abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = 0.0
    # Interpolate predicted depth to ground-truth resolution
    pred = interpolate_image(
        pred, gt.shape, mode='bilinear', align_corners=True)
    pred = pred.clamp(min=1e-6)
    # If using crop
    if config.crop == 'garg':
        crop = True
        crop_mask = torch.zeros(gt.shape[-2:]).byte().type_as(gt)
        y1, y2 = int(0.40810811 * gt_height), int(0.99189189 * gt_height)
        x1, x2 = int(0.03594771 * gt_width), int(0.96405229 * gt_width)
        crop_mask[y1:y2, x1:x2] = 1
    if config.crop == 'eigen_nyu':
        crop = True
        crop_mask = torch.zeros(gt.shape[-2:]).byte().type_as(gt)
        y1, y2 = 20, 459
        x1, x2 = 24, 615
        crop_mask[y1:y2, x1:x2] = 1
    # For each depth map
    for pred_i, gt_i in zip(pred, gt):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > config.min_depth) & (gt_i < config.max_depth)
        valid = valid & crop_mask.bool() if crop else valid
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]
        # Ground-truth median scaling if needed
        if use_gt_scale:
            pred_i = pred_i * torch.median(gt_i / pred_i)
            pred_i = torch.clamp(pred_i, config.min_depth, config.max_depth)

        # Clamp predicted depth values to min/max values
        pred_i = pred_i.clamp(config.min_depth, config.max_depth)

        # Calculate depth metrics

        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25**2).float().mean()
        a3 += (thresh < 1.25**3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i**2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i**2))
        rmse_log += torch.sqrt(
            torch.mean((torch.log(gt_i) - torch.log(pred_i))**2))
        iabs_diff += torch.mean(torch.abs(1.0 / pred_i - 1.0 / gt_i))
        irmse += torch.sqrt(torch.mean((1.0 / gt_i - 1.0 / pred_i)**2))
        d_i = gt_i.log() - pred_i.log()
        SILog += ((d_i**2).mean() - (d_i.sum())**2 / len(d_i)**2)**0.5
        SI += ((diff_i**2).mean() - (diff_i.sum())**2 / len(diff_i)**2)**0.5
    # Return average values for each metric
    return torch.tensor([
        metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, SILog, iabs_diff]
    ]).type_as(gt)


def compute_depth_metrics_demon(config, gt, gt_pose, pred, use_gt_scale=True):
    # Initialize variables
    batch_size, _, gt_height, gt_width = gt.shape
    SI = SILog = iabs_diff = irmse = abs_diff = abs_rel = sq_rel = rmse = rmse_log = a1 = a2 = a3 = 0.0
    # Interpolate predicted depth to ground-truth resolution
    pred = interpolate_image(
        pred, gt.shape, mode='bilinear', align_corners=True)
    pred = pred.clamp(min=1e-6)
    # For each depth map
    for pred_i, gt_i, gt_pose_i in zip(pred, gt, gt_pose):
        gt_i, pred_i = torch.squeeze(gt_i), torch.squeeze(pred_i)
        # Keep valid pixels (min/max depth and crop)
        valid = (gt_i > config.min_depth) & (gt_i < config.max_depth)
        # Stop if there are no remaining valid pixels
        if valid.sum() == 0:
            continue
        # Keep only valid pixels
        gt_i, pred_i = gt_i[valid], pred_i[valid]

        # Ground-truth median scaling if needed
        if use_gt_scale:
            # gt normalization
            translation_gt = gt_pose_i[:, :3, 3]
            translation_norm = torch.sqrt(translation_gt[0].dot(
                translation_gt[0]))
            gt_i = gt_i / translation_norm

            pred_i = pred_i * torch.median(gt_i / pred_i)
            # d1d1 = pred_i * pred_i
            # d1d2 = pred_i * gt_i
            # mask = d1d2 > 0
            # sum_d1d1 = d1d1[mask].sum()
            # sum_d1d2 = d1d2[mask].sum()
            # scale = sum_d1d2/sum_d1d1
            # pred_i = pred_i * scale

        # Calculate depth metrics
        thresh = torch.max((gt_i / pred_i), (pred_i / gt_i))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25**2).float().mean()
        a3 += (thresh < 1.25**3).float().mean()

        diff_i = gt_i - pred_i
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / gt_i)
        sq_rel += torch.mean(diff_i**2 / gt_i)
        rmse += torch.sqrt(torch.mean(diff_i**2))
        rmse_log += torch.sqrt(
            torch.mean((torch.log(gt_i) - torch.log(pred_i))**2))
        iabs_diff += torch.mean(torch.abs(1.0 / pred_i - 1.0 / gt_i))
        irmse += torch.sqrt(torch.mean((1.0 / gt_i - 1.0 / pred_i)**2))
        d_i = gt_i.log() - pred_i.log()
        SILog += ((d_i**2).mean() - (d_i.sum())**2 / len(d_i)**2)**0.5
        SI += ((diff_i**2).mean() - (diff_i.sum())**2 / len(diff_i)**2)**0.5
    # Return average values for each metric
    return torch.tensor([
        metric / batch_size for metric in
        [abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3, SILog, iabs_diff]
    ]).type_as(gt)


def compute_pose_metrics(config, gt, pred):
    pr = pred[0].mat.squeeze().detach().cpu().numpy()
    gt = gt[0].squeeze().detach().cpu().numpy()

    # seperate rotations and translations
    R1, t1 = gt[:3, :3], gt[:3, 3]
    R2, t2 = pr[:3, :3], pr[:3, 3]

    costheta = (np.trace(np.dot(R1.T, R2)) - 1.0) / 2.0
    costheta = np.minimum(costheta, 1.0)
    rdeg = np.arccos(costheta) * (180 / np.pi)

    t1mag = np.sqrt(np.dot(t1, t1))
    t2mag = np.sqrt(np.dot(t2, t2))
    costheta = np.dot(t1, t2) / (t1mag * t2mag)
    tdeg = np.arccos(costheta) * (180 / np.pi)

    # fit scales to translations
    a = np.dot(t1, t2) / np.dot(t2, t2)
    tcm = 100 * np.sqrt(np.sum((t1 - a * t2)**2, axis=-1))

    return torch.Tensor([rdeg, tdeg, tcm])
