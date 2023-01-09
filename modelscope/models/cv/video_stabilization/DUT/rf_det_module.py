# @Time    : 2018-9-27 15:39
# @Author  : xylon
# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_stabilization.utils.image_utils import (
    filter_border, get_gauss_filter_weight, nms, topk_map)


class RFDetModule(nn.Module):

    def __init__(
        self,
        score_com_strength,
        scale_com_strength,
        nms_thresh,
        nms_ksize,
        topk,
        gauss_ksize,
        gauss_sigma,
        ksize,
        padding,
        dilation,
        scale_list,
    ):
        super(RFDetModule, self).__init__()

        self.score_com_strength = score_com_strength
        self.scale_com_strength = scale_com_strength
        self.NMS_THRESH = nms_thresh
        self.NMS_KSIZE = nms_ksize
        self.TOPK = topk
        self.GAUSSIAN_KSIZE = gauss_ksize
        self.GAUSSIAN_SIGMA = gauss_sigma

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 3 RF
        self.insnorm1 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s3 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s3 = nn.InstanceNorm2d(1, affine=True)

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 5 RF
        self.insnorm2 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s5 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s5 = nn.InstanceNorm2d(1, affine=True)

        self.conv3 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 7 RF
        self.insnorm3 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s7 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s7 = nn.InstanceNorm2d(1, affine=True)

        self.conv4 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 9 RF
        self.insnorm4 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s9 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s9 = nn.InstanceNorm2d(1, affine=True)

        self.conv5 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 11 RF
        self.insnorm5 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s11 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s11 = nn.InstanceNorm2d(1, affine=True)

        self.conv6 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 13 RF
        self.insnorm6 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s13 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s13 = nn.InstanceNorm2d(1, affine=True)

        self.conv7 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 15 RF
        self.insnorm7 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s15 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s15 = nn.InstanceNorm2d(1, affine=True)

        self.conv8 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 17 RF
        self.insnorm8 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s17 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s17 = nn.InstanceNorm2d(1, affine=True)

        self.conv9 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 19 RF
        self.insnorm9 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s19 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s19 = nn.InstanceNorm2d(1, affine=True)

        self.conv10 = nn.Conv2d(
            in_channels=16,
            out_channels=16,
            kernel_size=ksize,
            stride=1,
            padding=padding,
            dilation=dilation,
        )  # 21 RF
        self.insnorm10 = nn.InstanceNorm2d(16, affine=True)
        self.conv_s21 = nn.Conv2d(
            in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.insnorm_s21 = nn.InstanceNorm2d(1, affine=True)

        self.scale_list = torch.tensor(scale_list)

    def forward(self, **kwargs):
        pass

    def process(self, im1w_score):
        """
        nms(n), topk(t), gaussian kernel(g) operation
        :param im1w_score: warped score map
        :return: processed score map, topk mask, topk value
        """
        im1w_score = filter_border(im1w_score)

        # apply nms to im1w_score
        nms_mask = nms(
            im1w_score, thresh=self.NMS_THRESH, ksize=self.NMS_KSIZE)
        im1w_score = im1w_score * nms_mask
        topk_value = im1w_score

        # apply topk to im1w_score
        topk_mask = topk_map(im1w_score, self.TOPK)
        im1w_score = topk_mask.to(torch.float) * im1w_score

        # apply gaussian kernel to im1w_score
        psf = get_gauss_filter_weight(
            self.GAUSSIAN_KSIZE,
            self.GAUSSIAN_SIGMA)[None, None, :, :].clone().detach().to(
                im1w_score.device)
        im1w_score = F.conv2d(
            input=im1w_score.permute(0, 3, 1, 2),
            weight=psf,
            stride=1,
            padding=self.GAUSSIAN_KSIZE // 2,
        ).permute(0, 2, 3, 1)  # (B, H, W, 1)
        """
        apply tf.clamp to make sure all value in im1w_score isn't greater than 1
        but this won't happend in correct way
        """
        im1w_score = im1w_score.clamp(min=0.0, max=1.0)

        return im1w_score, topk_mask, topk_value

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(
                m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
            try:
                nn.init.xavier_uniform_(m.bias.data)
            except Exception:
                pass
