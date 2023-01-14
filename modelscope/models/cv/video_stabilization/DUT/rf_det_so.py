# @Time    : 2018-9-13 16:03
# @Author  : xylon
# Part of the implementation is borrowed and modified from DUTCode,
# publicly available at https://github.com/Annbless/DUTCode

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_stabilization.utils.image_utils import (
    soft_max_and_argmax_1d, soft_nms_3d)
from modelscope.models.cv.video_stabilization.utils.math_utils import L2Norm
from .rf_det_module import RFDetModule


class RFDetSO(RFDetModule):

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
        super(RFDetSO, self).__init__(
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
        )

        self.conv_o3 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o5 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o7 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o9 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o11 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o13 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o15 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o17 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o19 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)
        self.conv_o21 = nn.Conv2d(
            in_channels=16, out_channels=2, kernel_size=1, stride=1, padding=0)

    def forward(self, photos):

        # Extract score map in scale space from 3 to 21
        score_featmaps_s3 = F.leaky_relu(self.insnorm1(self.conv1(photos)))
        score_map_s3 = self.insnorm_s3(
            self.conv_s3(score_featmaps_s3)).permute(0, 2, 3, 1)
        orint_map_s3 = (
            L2Norm(self.conv_o3(score_featmaps_s3),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))

        score_featmaps_s5 = F.leaky_relu(
            self.insnorm2(self.conv2(score_featmaps_s3)))
        score_map_s5 = self.insnorm_s5(
            self.conv_s5(score_featmaps_s5)).permute(0, 2, 3, 1)
        orint_map_s5 = (
            L2Norm(self.conv_o5(score_featmaps_s5),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s5 = score_featmaps_s5 + score_featmaps_s3

        score_featmaps_s7 = F.leaky_relu(
            self.insnorm3(self.conv3(score_featmaps_s5)))
        score_map_s7 = self.insnorm_s7(
            self.conv_s7(score_featmaps_s7)).permute(0, 2, 3, 1)
        orint_map_s7 = (
            L2Norm(self.conv_o7(score_featmaps_s7),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s7 = score_featmaps_s7 + score_featmaps_s5

        score_featmaps_s9 = F.leaky_relu(
            self.insnorm4(self.conv4(score_featmaps_s7)))
        score_map_s9 = self.insnorm_s9(
            self.conv_s9(score_featmaps_s9)).permute(0, 2, 3, 1)
        orint_map_s9 = (
            L2Norm(self.conv_o9(score_featmaps_s9),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s9 = score_featmaps_s9 + score_featmaps_s7

        score_featmaps_s11 = F.leaky_relu(
            self.insnorm5(self.conv5(score_featmaps_s9)))
        score_map_s11 = self.insnorm_s11(
            self.conv_s11(score_featmaps_s11)).permute(0, 2, 3, 1)
        orint_map_s11 = (
            L2Norm(self.conv_o11(score_featmaps_s11),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s11 = score_featmaps_s11 + score_featmaps_s9

        score_featmaps_s13 = F.leaky_relu(
            self.insnorm6(self.conv6(score_featmaps_s11)))
        score_map_s13 = self.insnorm_s13(
            self.conv_s13(score_featmaps_s13)).permute(0, 2, 3, 1)
        orint_map_s13 = (
            L2Norm(self.conv_o13(score_featmaps_s13),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s13 = score_featmaps_s13 + score_featmaps_s11

        score_featmaps_s15 = F.leaky_relu(
            self.insnorm7(self.conv7(score_featmaps_s13)))
        score_map_s15 = self.insnorm_s15(
            self.conv_s15(score_featmaps_s15)).permute(0, 2, 3, 1)
        orint_map_s15 = (
            L2Norm(self.conv_o15(score_featmaps_s15),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s15 = score_featmaps_s15 + score_featmaps_s13

        score_featmaps_s17 = F.leaky_relu(
            self.insnorm8(self.conv8(score_featmaps_s15)))
        score_map_s17 = self.insnorm_s17(
            self.conv_s17(score_featmaps_s17)).permute(0, 2, 3, 1)
        orint_map_s17 = (
            L2Norm(self.conv_o17(score_featmaps_s17),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s17 = score_featmaps_s17 + score_featmaps_s15

        score_featmaps_s19 = F.leaky_relu(
            self.insnorm9(self.conv9(score_featmaps_s17)))
        score_map_s19 = self.insnorm_s19(
            self.conv_s19(score_featmaps_s19)).permute(0, 2, 3, 1)
        orint_map_s19 = (
            L2Norm(self.conv_o19(score_featmaps_s19),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))
        score_featmaps_s19 = score_featmaps_s19 + score_featmaps_s17

        score_featmaps_s21 = F.leaky_relu(
            self.insnorm10(self.conv10(score_featmaps_s19)))
        score_map_s21 = self.insnorm_s21(
            self.conv_s21(score_featmaps_s21)).permute(0, 2, 3, 1)
        orint_map_s21 = (
            L2Norm(self.conv_o21(score_featmaps_s21),
                   dim=1).permute(0, 2, 3, 1).unsqueeze(-2))

        score_maps = torch.cat(
            (
                score_map_s3,
                score_map_s5,
                score_map_s7,
                score_map_s9,
                score_map_s11,
                score_map_s13,
                score_map_s15,
                score_map_s17,
                score_map_s19,
                score_map_s21,
            ),
            -1,
        )  # (B, H, W, C)

        orint_maps = torch.cat(
            (
                orint_map_s3,
                orint_map_s5,
                orint_map_s7,
                orint_map_s9,
                orint_map_s11,
                orint_map_s13,
                orint_map_s15,
                orint_map_s17,
                orint_map_s19,
                orint_map_s21,
            ),
            -2,
        )  # (B, H, W, 10, 2)

        # get each pixel probability in all scale
        scale_probs = soft_nms_3d(score_maps, ksize=15, com_strength=3.0)

        # get each pixel probability summary from all scale space and correspond scale value
        score_map, scale_map, orint_map = soft_max_and_argmax_1d(
            input=scale_probs,
            orint_maps=orint_maps,
            dim=-1,
            scale_list=self.scale_list,
            keepdim=True,
            com_strength1=self.score_com_strength,
            com_strength2=self.scale_com_strength,
        )

        return score_map, scale_map, orint_map

    @staticmethod
    def convO_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight.data)
            try:
                nn.init.ones_(m.bias.data)
            except Exception:
                pass
