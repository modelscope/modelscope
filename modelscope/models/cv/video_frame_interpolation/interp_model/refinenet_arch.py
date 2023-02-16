# Part of the implementation is borrowed and modified from QVI, publicly available at https://github.com/xuxy09/QVI

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_frame_interpolation.interp_model.flow_reversal import \
    FlowReversal
from modelscope.models.cv.video_frame_interpolation.interp_model.IFNet_swin import \
    IFNet
from modelscope.models.cv.video_frame_interpolation.interp_model.UNet import \
    Small_UNet_Ds


class AcFusionLayer(nn.Module):

    def __init__(self, ):
        super(AcFusionLayer, self).__init__()

    def forward(self, flo10, flo12, flo21, flo23, t=0.5):
        return 0.5 * ((t + t**2) * flo12 - (t - t**2) * flo10), \
            0.5 * (((1 - t) + (1 - t)**2) * flo21 - ((1 - t) - (1 - t)**2) * flo23)
        # return 0.375 * flo12 - 0.125 * flo10, 0.375 * flo21 - 0.125 * flo23


class Get_gradient(nn.Module):

    def __init__(self):
        super(Get_gradient, self).__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]  # R
        x1 = x[:, 1]  # G
        x2 = x[:, 2]  # B
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class LowPassFilter(nn.Module):

    def __init__(self):
        super(LowPassFilter, self).__init__()
        kernel_lpf = [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1]]

        kernel_lpf = torch.FloatTensor(kernel_lpf).unsqueeze(0).unsqueeze(
            0) / 49

        self.weight_lpf = nn.Parameter(data=kernel_lpf, requires_grad=False)

    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        y0 = F.conv2d(x0.unsqueeze(1), self.weight_lpf, padding=3)
        y1 = F.conv2d(x1.unsqueeze(1), self.weight_lpf, padding=3)

        y = torch.cat([y0, y1], dim=1)

        return y


def backwarp(img, flow):
    _, _, H, W = img.size()

    u = flow[:, 0, :, :]
    v = flow[:, 1, :, :]

    gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))

    gridX = torch.tensor(
        gridX,
        requires_grad=False,
    ).cuda()
    gridY = torch.tensor(
        gridY,
        requires_grad=False,
    ).cuda()
    x = gridX.unsqueeze(0).expand_as(u).float() + u
    y = gridY.unsqueeze(0).expand_as(v).float() + v

    x = 2 * (x / (W - 1) - 0.5)
    y = 2 * (y / (H - 1) - 0.5)

    grid = torch.stack((x, y), dim=3)

    imgOut = torch.nn.functional.grid_sample(
        img, grid, padding_mode='border', align_corners=True)

    return imgOut


class SmallMaskNet(nn.Module):
    """A three-layer network for predicting mask"""

    def __init__(self, input, output):
        super(SmallMaskNet, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, output, 3, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.1)
        x = self.conv3(x)
        return x


class StaticMaskNet(nn.Module):
    """static mask"""

    def __init__(self, input, output):
        super(StaticMaskNet, self).__init__()

        modules_body = []
        modules_body.append(
            nn.Conv2d(
                in_channels=input,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1))
        modules_body.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))
        modules_body.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1))
        modules_body.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))
        modules_body.append(
            nn.Conv2d(
                in_channels=32,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1))
        modules_body.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))
        modules_body.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=1))
        modules_body.append(nn.LeakyReLU(inplace=False, negative_slope=0.1))
        modules_body.append(
            nn.Conv2d(
                in_channels=16,
                out_channels=output,
                kernel_size=3,
                stride=1,
                padding=1))
        modules_body.append(nn.Sigmoid())

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        y = self.body(x)
        return y


def tensor_erode(bin_img, ksize=5):
    B, C, H, W = bin_img.shape
    pad = (ksize - 1) // 2
    bin_img = F.pad(bin_img, [pad, pad, pad, pad], mode='constant', value=0)

    patches = bin_img.unfold(dimension=2, size=ksize, step=1)
    patches = patches.unfold(dimension=3, size=ksize, step=1)

    eroded, _ = patches.reshape(B, C, H, W, -1).max(dim=-1)
    return eroded


class QVI_inter_Ds(nn.Module):
    """Given flow, implement Quadratic Video Interpolation"""

    def __init__(self, debug_en=False, is_training=False):
        super(QVI_inter_Ds, self).__init__()
        self.acc = AcFusionLayer()
        self.fwarp = FlowReversal()
        self.refinenet = Small_UNet_Ds(20, 8)
        self.masknet = SmallMaskNet(38, 1)

        self.staticnet = StaticMaskNet(56, 1)
        self.lpfilter = LowPassFilter()

        self.get_grad = Get_gradient()
        self.debug_en = debug_en
        self.is_training = is_training

    def fill_flow_hole(self, ft, norm, ft_fill):
        (N, C, H, W) = ft.shape
        ft[norm == 0] = ft_fill[norm == 0]

        ft_1 = self.lpfilter(ft.clone())
        ft_ds = torch.nn.functional.interpolate(
            input=ft_1,
            size=(H // 4, W // 4),
            mode='bilinear',
            align_corners=False)
        ft_up = torch.nn.functional.interpolate(
            input=ft_ds, size=(H, W), mode='bilinear', align_corners=False)

        ft[norm == 0] = ft_up[norm == 0]

        return ft

    def forward(self, F10_Ds, F12_Ds, F21_Ds, F23_Ds, I1_Ds, I2_Ds, I1, I2, t):
        if F12_Ds is None or F21_Ds is None:
            return I1

        if F10_Ds is not None and F23_Ds is not None:
            F1t_Ds, F2t_Ds = self.acc(F10_Ds, F12_Ds, F21_Ds, F23_Ds, t)

        else:
            F1t_Ds = t * F12_Ds
            F2t_Ds = (1 - t) * F21_Ds

        # Flow Reversal
        F1t_Ds2 = F.interpolate(
            F1t_Ds, scale_factor=1.0 / 3, mode='nearest') / 3
        F2t_Ds2 = F.interpolate(
            F2t_Ds, scale_factor=1.0 / 3, mode='nearest') / 3
        Ft1_Ds2, norm1_Ds2 = self.fwarp(F1t_Ds2, F1t_Ds2)
        Ft1_Ds2 = -Ft1_Ds2
        Ft2_Ds2, norm2_Ds2 = self.fwarp(F2t_Ds2, F2t_Ds2)
        Ft2_Ds2 = -Ft2_Ds2

        Ft1_Ds2[norm1_Ds2 > 0] \
            = Ft1_Ds2[norm1_Ds2 > 0] / norm1_Ds2[norm1_Ds2 > 0].clone()
        Ft2_Ds2[norm2_Ds2 > 0] \
            = Ft2_Ds2[norm2_Ds2 > 0] / norm2_Ds2[norm2_Ds2 > 0].clone()
        if 1:
            Ft1_Ds2_fill = -F1t_Ds2
            Ft2_Ds2_fill = -F2t_Ds2
            Ft1_Ds2 = self.fill_flow_hole(Ft1_Ds2, norm1_Ds2, Ft1_Ds2_fill)
            Ft2_Ds2 = self.fill_flow_hole(Ft2_Ds2, norm2_Ds2, Ft2_Ds2_fill)

        Ft1_Ds = F.interpolate(
            Ft1_Ds2, size=[F1t_Ds.size(2), F1t_Ds.size(3)], mode='nearest') * 3
        Ft2_Ds = F.interpolate(
            Ft2_Ds2, size=[F2t_Ds.size(2), F2t_Ds.size(3)], mode='nearest') * 3

        I1t_Ds = backwarp(I1_Ds, Ft1_Ds)
        I2t_Ds = backwarp(I2_Ds, Ft2_Ds)

        output_Ds, feature_Ds = self.refinenet(
            torch.cat(
                [I1_Ds, I2_Ds, I1t_Ds, I2t_Ds, F12_Ds, F21_Ds, Ft1_Ds, Ft2_Ds],
                dim=1))

        # Adaptive filtering
        Ft1r_Ds = backwarp(
            Ft1_Ds, 10 * torch.tanh(output_Ds[:, 4:6])) + output_Ds[:, :2]
        Ft2r_Ds = backwarp(
            Ft2_Ds, 10 * torch.tanh(output_Ds[:, 6:8])) + output_Ds[:, 2:4]

        # warping and fusing
        I1tf_Ds = backwarp(I1_Ds, Ft1r_Ds)
        I2tf_Ds = backwarp(I2_Ds, Ft2r_Ds)

        G1_Ds = self.get_grad(I1_Ds)
        G2_Ds = self.get_grad(I2_Ds)
        G1tf_Ds = backwarp(G1_Ds, Ft1r_Ds)
        G2tf_Ds = backwarp(G2_Ds, Ft2r_Ds)

        M_Ds = torch.sigmoid(
            self.masknet(torch.cat([I1tf_Ds, I2tf_Ds, feature_Ds],
                                   dim=1))).repeat(1, 3, 1, 1)

        Ft1r = F.interpolate(
            Ft1r_Ds * 2, scale_factor=2, mode='bilinear', align_corners=False)
        Ft2r = F.interpolate(
            Ft2r_Ds * 2, scale_factor=2, mode='bilinear', align_corners=False)

        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = F.interpolate(
            M_Ds, scale_factor=2, mode='bilinear', align_corners=False)

        # fuse
        It_warp = ((1 - t) * M * I1tf + t * (1 - M) * I2tf) \
            / ((1 - t) * M + t * (1 - M)).clone()

        # static blending
        It_static = (1 - t) * I1 + t * I2
        tmp = torch.cat((I1tf_Ds, I2tf_Ds, G1tf_Ds, G2tf_Ds, I1_Ds, I2_Ds,
                         G1_Ds, G2_Ds, feature_Ds),
                        dim=1)
        M_static_Ds = self.staticnet(tmp)
        M_static_dilate = tensor_erode(M_static_Ds)
        M_static_dilate = tensor_erode(M_static_dilate)
        M_static = F.interpolate(
            M_static_dilate,
            scale_factor=2,
            mode='bilinear',
            align_corners=False)

        It_warp = (1 - M_static) * It_warp + M_static * It_static

        if self.is_training:
            return It_warp, Ft1r, Ft2r
        else:
            if self.debug_en:
                return It_warp, M, M_static, I1tf, I2tf, Ft1r, Ft2r
            else:
                return It_warp


class QVI_inter(nn.Module):
    """Given flow, implement Quadratic Video Interpolation"""

    def __init__(self, debug_en=False, is_training=False):
        super(QVI_inter, self).__init__()
        self.acc = AcFusionLayer()
        self.fwarp = FlowReversal()
        self.refinenet = Small_UNet_Ds(20, 8)
        self.masknet = SmallMaskNet(38, 1)

        self.staticnet = StaticMaskNet(56, 1)
        self.lpfilter = LowPassFilter()

        self.get_grad = Get_gradient()
        self.debug_en = debug_en
        self.is_training = is_training

    def fill_flow_hole(self, ft, norm, ft_fill):
        (N, C, H, W) = ft.shape
        ft[norm == 0] = ft_fill[norm == 0]

        ft_1 = self.lpfilter(ft.clone())
        ft_ds = torch.nn.functional.interpolate(
            input=ft_1,
            size=(H // 4, W // 4),
            mode='bilinear',
            align_corners=False)
        ft_up = torch.nn.functional.interpolate(
            input=ft_ds, size=(H, W), mode='bilinear', align_corners=False)

        ft[norm == 0] = ft_up[norm == 0]

        return ft

    def forward(self, F10, F12, F21, F23, I1, I2, t):
        if F12 is None or F21 is None:
            return I1

        if F10 is not None and F23 is not None:
            F1t, F2t = self.acc(F10, F12, F21, F23, t)

        else:
            F1t = t * F12
            F2t = (1 - t) * F21

        # Flow Reversal
        F1t_Ds = F.interpolate(F1t, scale_factor=1.0 / 3, mode='nearest') / 3
        F2t_Ds = F.interpolate(F2t, scale_factor=1.0 / 3, mode='nearest') / 3
        Ft1_Ds, norm1_Ds = self.fwarp(F1t_Ds, F1t_Ds)
        Ft1_Ds = -Ft1_Ds
        Ft2_Ds, norm2_Ds = self.fwarp(F2t_Ds, F2t_Ds)
        Ft2_Ds = -Ft2_Ds

        Ft1_Ds[norm1_Ds > 0] \
            = Ft1_Ds[norm1_Ds > 0] / norm1_Ds[norm1_Ds > 0].clone()
        Ft2_Ds[norm2_Ds > 0] \
            = Ft2_Ds[norm2_Ds > 0] / norm2_Ds[norm2_Ds > 0].clone()
        if 1:
            Ft1_fill = -F1t_Ds
            Ft2_fill = -F2t_Ds
            Ft1_Ds = self.fill_flow_hole(Ft1_Ds, norm1_Ds, Ft1_fill)
            Ft2_Ds = self.fill_flow_hole(Ft2_Ds, norm2_Ds, Ft2_fill)

        Ft1 = F.interpolate(
            Ft1_Ds, size=[F1t.size(2), F1t.size(3)], mode='nearest') * 3
        Ft2 = F.interpolate(
            Ft2_Ds, size=[F2t.size(2), F2t.size(3)], mode='nearest') * 3

        I1t = backwarp(I1, Ft1)
        I2t = backwarp(I2, Ft2)

        output, feature = self.refinenet(
            torch.cat([I1, I2, I1t, I2t, F12, F21, Ft1, Ft2], dim=1))

        # Adaptive filtering
        Ft1r = backwarp(Ft1, 10 * torch.tanh(output[:, 4:6])) + output[:, :2]
        Ft2r = backwarp(Ft2, 10 * torch.tanh(output[:, 6:8])) + output[:, 2:4]

        # warping and fusing
        I1tf = backwarp(I1, Ft1r)
        I2tf = backwarp(I2, Ft2r)

        M = torch.sigmoid(
            self.masknet(torch.cat([I1tf, I2tf, feature],
                                   dim=1))).repeat(1, 3, 1, 1)

        It_warp = ((1 - t) * M * I1tf + t * (1 - M) * I2tf) \
            / ((1 - t) * M + t * (1 - M)).clone()

        G1 = self.get_grad(I1)
        G2 = self.get_grad(I2)
        G1tf = backwarp(G1, Ft1r)
        G2tf = backwarp(G2, Ft2r)

        # static blending
        It_static = (1 - t) * I1 + t * I2
        M_static = self.staticnet(
            torch.cat([I1tf, I2tf, G1tf, G2tf, I1, I2, G1, G2, feature],
                      dim=1))
        M_static_dilate = tensor_erode(M_static)
        M_static_dilate = tensor_erode(M_static_dilate)
        It_warp = (1 - M_static_dilate) * It_warp + M_static_dilate * It_static

        if self.is_training:
            return It_warp, Ft1r, Ft2r
        else:
            if self.debug_en:
                return It_warp, M, M_static, I1tf, I2tf, Ft1r, Ft2r
            else:
                return It_warp


class InterpNetDs(nn.Module):

    def __init__(self, debug_en=False, is_training=False):
        super(InterpNetDs, self).__init__()
        self.ifnet = IFNet()
        self.internet = QVI_inter_Ds(
            debug_en=debug_en, is_training=is_training)

    def forward(self,
                img1,
                img2,
                F10_up,
                F12_up,
                F21_up,
                F23_up,
                UHD=2,
                timestep=0.5):
        F12, F21 = self.ifnet(img1, img2, F12_up, F21_up, UHD)
        It_warp = self.internet(F10_up, F12, F21, F23_up, img1, img2, timestep)

        return It_warp


class InterpNet(nn.Module):

    def __init__(self, debug_en=False, is_training=False):
        super(InterpNet, self).__init__()
        self.ifnet = IFNet()
        self.internet = QVI_inter(debug_en=debug_en, is_training=is_training)

    def forward(self,
                img1,
                img2,
                F10_up,
                F12_up,
                F21_up,
                F23_up,
                UHD=2,
                timestep=0.5):
        F12, F21 = self.ifnet(img1, img2, F12_up, F21_up, UHD)
        It_warp = self.internet(F10_up, F12, F21, F23_up, img1, img2, timestep)

        return It_warp
