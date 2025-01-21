# @Description: Basic implementation of Frequency Domain Filtering strategy (Sec 3.2 in the paper).
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07
# @https://github.com/doublez0108/geomvsnet

import matplotlib.pyplot as plt
import numpy as np
import torch


def frequency_domain_filter(depth, rho_ratio):
    """
    large rho_ratio -> more information filtered
    """
    f = torch.fft.fft2(depth)
    fshift = torch.fft.fftshift(f)

    b, h, w = depth.shape
    k_h, k_w = h / rho_ratio, w / rho_ratio

    fshift[:, :int(h / 2 - k_h / 2), :] = 0
    fshift[:, int(h / 2 + k_h / 2):, :] = 0
    fshift[:, :, :int(w / 2 - k_w / 2)] = 0
    fshift[:, :, int(w / 2 + k_w / 2):] = 0

    ishift = torch.fft.ifftshift(fshift)
    idepth = torch.fft.ifft2(ishift)
    depth_filtered = torch.abs(idepth)

    return depth_filtered


def visual_fft_fig(fshift):
    fft_fig = torch.abs(20 * torch.log(fshift))
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(fft_fig[0, :, :], cmap='gray')
