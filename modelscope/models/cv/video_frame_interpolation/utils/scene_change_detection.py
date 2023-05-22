# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_hist(img_tensor):
    hist = torch.histc(img_tensor, bins=64, min=0, max=255)
    return hist / img_tensor.numel()


def do_scene_detect(F01_tensor, F10_tensor, img0_tensor, img1_tensor):
    device = img0_tensor.device
    scene_change = False
    img0_tensor = img0_tensor.clone()
    img1_tensor = img1_tensor.clone()

    img0_gray = 0.299 * img0_tensor[:, 0:
                                    1] + 0.587 * img0_tensor[:, 1:
                                                             2] + 0.114 * img0_tensor[:,
                                                                                      2:
                                                                                      3]
    img1_gray = 0.299 * img1_tensor[:, 0:
                                    1] + 0.587 * img1_tensor[:, 1:
                                                             2] + 0.114 * img1_tensor[:,
                                                                                      2:
                                                                                      3]
    img0_gray = torch.clamp(img0_gray, 0, 255).byte().float().cpu()
    img1_gray = torch.clamp(img1_gray, 0, 255).byte().float().cpu()

    hist0 = calc_hist(img0_gray)
    hist1 = calc_hist(img1_gray)
    diff = torch.abs(hist0 - hist1)
    diff[diff < 0.01] = 0
    if torch.sum(diff) > 0.8 or diff.max() > 0.4:
        return True
    img0_gray = img0_gray.to(device)
    img1_gray = img1_gray.to(device)

    # second stage: detect mv and pix mismatch

    (n, c, h, w) = F01_tensor.size()
    scale_x = w / 1920
    scale_y = h / 1080

    # compare mv
    (y, x) = torch.meshgrid(torch.arange(h), torch.arange(w))
    (y_grid, x_grid) = torch.meshgrid(
        torch.arange(64, h - 64, 8), torch.arange(64, w - 64, 8))
    x = x.to(device)
    y = y.to(device)
    y_grid = y_grid.to(device)
    x_grid = x_grid.to(device)
    fx = F01_tensor[0, 0]
    fy = F01_tensor[0, 1]
    x_ = x.float() + fx
    y_ = y.float() + fy
    x_ = torch.clamp(x_ + 0.5, 0, w - 1).long()
    y_ = torch.clamp(y_ + 0.5, 0, h - 1).long()

    grid_fx = fx[y_grid, x_grid]
    grid_fy = fy[y_grid, x_grid]

    x_grid_ = x_[y_grid, x_grid]
    y_grid_ = y_[y_grid, x_grid]

    grid_fx_ = F10_tensor[0, 0, y_grid_, x_grid_]
    grid_fy_ = F10_tensor[0, 1, y_grid_, x_grid_]

    sum_x = grid_fx + grid_fx_
    sum_y = grid_fy + grid_fy_
    distance = torch.sqrt(sum_x**2 + sum_y**2)

    fx_len = torch.abs(grid_fx) * scale_x
    fy_len = torch.abs(grid_fy) * scale_y
    ori_len = torch.where(fx_len > fy_len, fx_len, fy_len)

    thres = torch.clamp(0.1 * ori_len + 4, 5, 14)

    # compare pix diff
    ori_img = img0_gray
    ref_img = img1_gray[:, :, y_, x_]

    img_diff = ori_img.float() - ref_img.float()
    img_diff = torch.abs(img_diff)

    kernel = np.ones([8, 8], float) / 64
    kernel = torch.FloatTensor(kernel).to(device).unsqueeze(0).unsqueeze(0)
    diff = F.conv2d(img_diff, kernel, padding=4)

    diff = diff[0, 0, y_grid, x_grid]

    index = (distance > thres) * (diff > 5)
    if index.sum().float() / distance.numel() > 0.5:
        scene_change = True
    return scene_change
