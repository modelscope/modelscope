# @Time    : 2018-9-21 14:36
# @Author  : xylon
# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import torch
from skimage import transform
from torch.nn import functional as F

from modelscope.models.cv.video_stabilization.utils.math_utils import L2Norm


def clip_patch(kpts_byxc, kpts_scale, kpts_ori, im_info, images, PSIZE):
    """
    clip patch from im_C, im_S, im_info, im_raw.
    :param kpts_byxc: tensor #(B*topk, 4): the 4 correspond to (b, y, x, 0) each element in it has length B*topk
    :param kpts_scale: tensor(B*topk): image scale value corresponding to topk keypoints in all batch
    :param kpts_ori: tensor(B*topk, 2): image orintation value corresponding to topk keypoints in all batch
    :param im_info: tensor (B, 2): a list contain rescale ratio sh and sw
    :param images: tensor(B, 1, H, W): like 960*720 gray image before image rescaled to 320*240
    :param PSIZE: should be cfg.PATCH.size
    :return: torch(B*topk, psize, psize): B*topk patch resized
    """
    assert kpts_byxc.size(0) == kpts_scale.size(0)
    out_width = out_height = PSIZE
    device = kpts_byxc.device
    B, C, im_height, im_width = images.size()
    num_kp = kpts_byxc.size(0)  # B*K
    max_y = int(im_height - 1)
    max_x = int(im_width - 1)
    y_t, x_t = torch.meshgrid([
        torch.linspace(-1, 1, out_height, dtype=torch.float, device=device),
        torch.linspace(-1, 1, out_width, dtype=torch.float, device=device),
    ])
    one_t = x_t.new_full(x_t.size(), fill_value=1)
    x_t = x_t.contiguous().view(-1)
    y_t = y_t.contiguous().view(-1)
    one_t = one_t.view(-1)
    grid = torch.stack((x_t, y_t, one_t))  # (3, out_width*out_height)
    grid = grid.view(-1)  # (3*out_width*out_height)
    grid = grid.repeat(num_kp)  # (numkp*3*out_width*out_height)
    # [num_kp, 3, 81] # this grid is designed to mask on keypoint from its left-up[-1, -1] to right-bottom[1, 1]
    grid = grid.view(num_kp, 3, -1)

    #
    # create 6D affine from scale and orientation
    # [s, 0, 0]   [cos, -sin, 0]
    # [0, s, 0] * [sin,  cos, 0]
    # [0, 0, 1]   [0,    0,   1]
    #
    thetas = torch.eye(
        2, 3, dtype=torch.float,
        device=device)  # [[ 1.,  0.,  0.],[ 0.,  1.,  0.]] (2, 3)
    thetas = thetas.unsqueeze(0).repeat(num_kp, 1, 1)  # (num_kp, 2, 3)
    im_info = im_info[:, 0].unsqueeze(-1)  # (B, 1)
    kpts_scale = kpts_scale.view(im_info.size(0), -1) / im_info  # (B, topk)
    kpts_scale = kpts_scale.view(-1) / 2.0  # (numkp)
    thetas = thetas * kpts_scale[:, None, None]
    ones = torch.tensor([[[0, 0, 1]]], dtype=torch.float,
                        device=device).repeat(num_kp, 1, 1)  # (numkp, 1, 1)
    thetas = torch.cat((thetas, ones), 1)  # (num_kp, 3, 3)
    # thetas like this
    # [sw, 0,  0]
    # [0,  sh, 0]
    # [0,  0,  1]

    if kpts_ori is not None:
        cos = kpts_ori[:, 0].unsqueeze(-1)  # [num_kp, 1]
        sin = kpts_ori[:, 1].unsqueeze(-1)  # [num_kp, 1]
        zeros = cos.new_full(cos.size(), fill_value=0)
        ones = cos.new_full(cos.size(), fill_value=1)
        R = torch.cat((cos, -sin, zeros, sin, cos, zeros, zeros, zeros, ones),
                      dim=-1)
        R = R.view(-1, 3, 3)
        thetas = torch.matmul(thetas, R)

    # Apply transformation to regular grid
    # [num_kp,3,3] * [num_kp,3,H*W] = [num_kp, 3, 81] # magnify grid to each keypoint scale
    T_g = torch.matmul(thetas, grid)
    x = T_g[:, 0, :]  # (numkp, 81)
    y = T_g[:, 1, :]  # (numkp, 81)

    # get each keypoint x
    kp_x_ofst = kpts_byxc[:, 2].view(B, -1).float() / im_info  # (B, topk)
    kp_x_ofst = kp_x_ofst.view(-1, 1)  # (numkp, 1) get each keypoint x
    # get each keypoint y
    kp_y_ofst = kpts_byxc[:, 1].view(B, -1).float() / im_info  # (B, topk)
    kp_y_ofst = kp_y_ofst.view(-1, 1)  # (numkp, 1) get each keypoint y

    # centerize on keypoints
    # [num_kp,81] + # [num_kp,1] # move grid center on each keypoint
    x = x + kp_x_ofst
    # [num_kp,81] + # [num_kp,1] # move grid center on each keypoint
    y = y + kp_y_ofst
    x = x.view(-1)  # [num_kp*81]
    y = y.view(-1)  # [num_kp*81]

    # interpolation
    x0 = x.floor().long()  # [num_kp*81]
    x1 = x0 + 1  # [num_kp*81]
    y0 = y.floor().long()  # [num_kp*81]
    y1 = y0 + 1  # [num_kp*81]

    x0 = x0.clamp(min=0, max=max_x)  # [num_kp*81]
    x1 = x1.clamp(min=0, max=max_x)  # [num_kp*81]
    y0 = y0.clamp(min=0, max=max_y)  # [num_kp*81]
    y1 = y1.clamp(min=0, max=max_y)  # [num_kp*81]

    dim2 = im_width
    dim1 = im_width * im_height
    batch_inds = kpts_byxc[:, 0].unsqueeze(
        -1)  # (num_kp, 1) get each keypoint batch number
    base = batch_inds.repeat(
        1, out_height * out_width
    )  # [num_kp, 81] # means batch indexes correspond to each grid pixel
    # [num_kp*81] # correspond to each grid pixel start index if all pixel flatten to a vector
    base = base.view(-1) * dim1
    base_y0 = (
        base + y0 * dim2
    )  # correspond each grid pixel y0 pixel if all pixel flatten to a vector
    base_y1 = (
        base + y1 * dim2
    )  # correspond each grid pixel y1 pixel if all pixel flatten to a vector
    idx_a = (
        base_y0 + x0
    )  # correspond left_up point pixel index if all pixel flatten to a vector
    idx_b = base_y1 + x0  # left-bottom pixel
    idx_c = base_y0 + x1  # right-up pixel
    idx_d = base_y1 + x1  # right-bottom pixel

    im_flat = images.view(-1)  # [B*height*width] # flatten all pixel

    # [num_kp*81] # get pixel value in index idx_a
    Ia = im_flat.gather(0, idx_a)
    # [num_kp*81] # get pixel value in index idx_b
    Ib = im_flat.gather(0, idx_b)
    # [num_kp*81] # get pixel value in index idx_c
    Ic = im_flat.gather(0, idx_c)
    # [num_kp*81] # get pixel value in index idx_d
    Id = im_flat.gather(0, idx_d)

    x0_f = x0.float()  # [num_kp*81]
    x1_f = x1.float()  # [num_kp*81]
    y0_f = y0.float()  # [num_kp*81]
    y1_f = y1.float()  # [num_kp*81]

    # [num_kp*81] # interpolation weight which is the distance from x to x1 times y to y1
    wa = (x1_f - x) * (y1_f - y)
    wb = (x1_f - x) * (y - y0_f)  # [num_kp*81] # interpolation weight
    wc = (x - x0_f) * (y1_f - y)  # [num_kp*81] # interpolation weight
    wd = (x - x0_f) * (y - y0_f)  # [num_kp*81] # interpolation weight

    output = (wa * Ia + wb * Ib + wc * Ic + wd * Id
              )  # interpolation value in each keypoints grid
    output = output.view(num_kp, out_height, out_width)
    return output.unsqueeze(1)


def warp(im1_data, homo21):
    """
    warp im1 to im2
    cause we get pixel valu ein im2 from im1
    so we warp grid in im2 to im1 that we need homo21
    :param im1_data: (B, H, W, C)
    :param homo21: (B, 3, 3)
    :return: out_image (B, H, W, C)
    """
    B, imH, imW, C = im1_data.size()
    outH, outW = imH, imW
    gy, gx = torch.meshgrid([torch.arange(outH), torch.arange(outW)])
    gx, gy = gx.float().unsqueeze(-1), gy.float().unsqueeze(-1)
    ones = gy.new_full(gy.size(), fill_value=1)
    grid = torch.cat((gx, gy, ones), -1)  # (H, W, 3)
    grid = grid.unsqueeze(0)  # (1, H, W, 3)
    grid = grid.repeat(B, 1, 1, 1)  # (B, H, W, 3)
    grid = grid.view(grid.size(0), -1, grid.size(-1))  # (B, H*W, 3)
    grid = grid.permute(0, 2, 1)  # (B, 3, H*W)
    grid = grid.type_as(homo21).to(homo21.device)

    # (B, 3, 3) matmul (B, 3, H*W) => (B, 3, H*W)
    grid_w = torch.matmul(homo21, grid)
    grid_w = grid_w.permute(0, 2, 1)  # (B, H*W, 3)
    grid_w = grid_w.div(grid_w[:, :, 2].unsqueeze(-1) + 1e-8)  # (B, H*W, 3)
    grid_w = grid_w.view(B, outH, outW, -1)[:, :, :, :2]  # (B, H, W, 2)
    grid_w[:, :, :, 0] = grid_w[:, :, :, 0].div(imW - 1) * 2 - 1
    grid_w[:, :, :, 1] = grid_w[:, :, :, 1].div(imH - 1) * 2 - 1

    out_image = torch.nn.functional.grid_sample(
        im1_data.permute(0, 3, 1, 2), grid_w)  # (B, C, H, W)

    return out_image.permute(0, 2, 3, 1)


def filtbordmask(imscore, radius):
    bs, height, width, c = imscore.size()
    mask = imscore.new_full((1, height - 2 * radius, width - 2 * radius, 1),
                            fill_value=1)
    mask = F.pad(
        input=mask,
        pad=(0, 0, radius, radius, radius, radius, 0, 0),
        mode='constant',
        value=0,
    )
    return mask


def filter_border(imscore, radius=8):
    imscore = imscore * filtbordmask(imscore, radius=radius)
    return imscore


def nms(input, thresh=0.0, ksize=5):
    """
    non maximum depression in each pixel if it is not maximum probability in its ksize*ksize range
    :param input: (B, H, W, 1)
    :param thresh: float
    :param ksize: int
    :return: mask (B, H, W, 1)
    """
    device = input.device
    batch, height, width, channel = input.size()
    pad = ksize // 2
    zeros = torch.zeros_like(input)
    input = torch.where(input < thresh, zeros, input)
    input_pad = F.pad(
        input=input,
        pad=(0, 0, 2 * pad, 2 * pad, 2 * pad, 2 * pad, 0, 0),
        mode='constant',
        value=0,
    )
    slice_map = torch.tensor([], dtype=input_pad.dtype, device=device)
    for i in range(ksize):
        for j in range(ksize):
            slice = input_pad[:, i:height + 2 * pad + i,
                              j:width + 2 * pad + j, :]
            slice_map = torch.cat((slice_map, slice), -1)

    max_slice = slice_map.max(dim=-1, keepdim=True)[0]
    center_map = slice_map[:, :, :, slice_map.size(-1) // 2].unsqueeze(-1)
    mask = torch.ge(center_map, max_slice)

    mask = mask[:, pad:height + pad, pad:width + pad, :]

    return mask.type_as(input)


def topk_map(maps, k=512):
    """
    find the top k maximum pixel probability in a maps
    :param maps: (B, H, W, 1)
    :param k: int
    :return: mask (B, H, W, 1)
    """
    batch, height, width, _ = maps.size()
    maps_flat = maps.view(batch, -1)

    indices = maps_flat.sort(dim=-1, descending=True)[1][:, :k]
    batch_idx = (
        torch.arange(0, batch, dtype=indices.dtype,
                     device=indices.device).unsqueeze(-1).repeat(1, k))
    batch_idx = batch_idx.view(-1).cpu().detach().numpy()
    row_idx = indices.contiguous().view(-1).cpu().detach().numpy()
    batch_indexes = (batch_idx, row_idx)

    topk_mask_flat = torch.zeros(
        maps_flat.size(), dtype=torch.uint8).to(maps.device)
    topk_mask_flat[batch_indexes] = 1

    mask = topk_mask_flat.view(batch, height, width, -1)
    return mask


def get_gauss_filter_weight(ksize, sig):
    """
    generate a gaussian kernel
    :param ksize: int
    :param sig: float
    :return: numpy(ksize*ksize)
    """
    mu_x = mu_y = ksize // 2
    if sig == 0:
        psf = torch.zeros((ksize, ksize)).float()
        psf[mu_y, mu_x] = 1.0
    else:
        sig = torch.tensor(sig).float()
        x = torch.arange(ksize)[None, :].repeat(ksize, 1).float()
        y = torch.arange(ksize)[:, None].repeat(1, ksize).float()
        psf = torch.exp(-(
            (x - mu_x)**2 / (2 * sig**2) + (y - mu_y)**2 / (2 * sig**2)))
    return psf


def soft_nms_3d(scale_logits, ksize, com_strength):
    """
    calculate probability for each pixel in each scale space
    :param scale_logits: (B, H, W, C)
    :param ksize: int
    :param com_strength: magnify parameter
    :return: probability for each pixel in each scale, size is (B, H, W, C)
    """
    num_scales = scale_logits.size(-1)

    max_each_scale = F.max_pool2d(
        input=scale_logits.permute(0, 3, 1, 2),
        kernel_size=ksize,
        padding=ksize // 2,
        stride=1,
    ).permute(0, 2, 3, 1)  # (B, H, W, C)
    max_all_scale, max_all_scale_idx = max_each_scale.max(
        dim=-1, keepdim=True)  # (B, H, W, 1)
    exp_maps = torch.exp(com_strength * (scale_logits - max_all_scale))
    sum_exp = F.conv2d(
        input=exp_maps.permute(0, 3, 1, 2).contiguous(),
        weight=exp_maps.new_full([1, num_scales, ksize, ksize],
                                 fill_value=1).contiguous(),
        stride=1,
        padding=ksize // 2,
    ).permute(0, 2, 3, 1)  # (B, H, W, 1)
    probs = exp_maps / (sum_exp + 1e-8)
    return probs


def soft_max_and_argmax_1d(input,
                           orint_maps,
                           scale_list,
                           com_strength1,
                           com_strength2,
                           dim=-1,
                           keepdim=True):
    """
    input should be pixel probability in each scale
    this function calculate the final pixel probability summary from all scale and each pixel correspond scale
    :param input: scale_probs(B, H, W, 10)
    :param orint_maps: (B, H, W, 10, 2)
    :param dim: final channel
    :param scale_list: scale space list
    :param keepdim: kepp dimension
    :param com_strength1: magnify argument of score
    :param com_strength2: magnify argument of scale
    :return: score_map(B, H, W, 1), scale_map(B, H, W, 1), (orint_map(B, H, W, 1, 2))
    """
    inputs_exp1 = torch.exp(
        com_strength1 * (input - torch.max(input, dim=dim, keepdim=True)[0]))
    input_softmax1 = inputs_exp1 / (
        inputs_exp1.sum(dim=dim, keepdim=True) + 1e-8)  # (B, H, W, 10)

    inputs_exp2 = torch.exp(
        com_strength2 * (input - torch.max(input, dim=dim, keepdim=True)[0]))
    input_softmax2 = inputs_exp2 / (
        inputs_exp2.sum(dim=dim, keepdim=True) + 1e-8)  # (B, H, W, 10)

    score_map = torch.sum(input * input_softmax1, dim=dim, keepdim=keepdim)

    scale_list_shape = [1] * len(input.size())
    scale_list_shape[dim] = -1
    scale_list = scale_list.view(scale_list_shape).to(input_softmax2.device)
    scale_map = torch.sum(
        scale_list * input_softmax2, dim=dim, keepdim=keepdim)

    if orint_maps is not None:
        orint_map = torch.sum(
            orint_maps * input_softmax1.unsqueeze(-1),
            dim=dim - 1,
            keepdim=keepdim)  # (B, H, W, 1, 2)
        orint_map = L2Norm(orint_map, dim=-1)
        return score_map, scale_map, orint_map
    else:
        return score_map, scale_map


def im_rescale(im, output_size):
    h, w = im.shape[:2]
    if isinstance(output_size, int):
        if h > w:
            new_h, new_w = output_size * h / w, output_size
        else:
            new_h, new_w = output_size, output_size * w / h
    else:
        new_h, new_w = output_size
    new_h, new_w = int(new_h), int(new_w)
    img = transform.resize(im, (new_h, new_w), mode='constant')

    return img, h, w, new_w / w, new_h / h
