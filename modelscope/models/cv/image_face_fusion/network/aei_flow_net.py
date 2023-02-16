# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .aad_layer import AAD_ResBlk
from .dense_motion import DenseMotionNetwork
from .ops import SpectralNorm, init_func


class Conv4x4(nn.Module):

    def __init__(self, in_c, out_c):
        super(Conv4x4, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, feat):
        x = self.conv(feat)
        x = self.norm(x)
        x = self.lrelu(x)
        return x


class DeConv4x4(nn.Module):

    def __init__(self, in_c, out_c):
        super(DeConv4x4, self).__init__()
        self.deconv = nn.ConvTranspose2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self, input, skip):
        x = self.deconv(input)
        x = self.bn(x)
        x = self.lrelu(x)
        return torch.cat((x, skip), dim=1)


class Attention(nn.Module):

    def __init__(self, ch, use_sn=True):
        super(Attention, self).__init__()
        self.ch = ch
        self.theta = nn.Conv2d(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(
            self.ch, self.ch // 8, kernel_size=1, padding=0, bias=False)
        self.g = nn.Conv2d(
            self.ch, self.ch // 2, kernel_size=1, padding=0, bias=False)
        self.o = nn.Conv2d(
            self.ch // 2, self.ch, kernel_size=1, padding=0, bias=False)
        if use_sn:
            self.theta = SpectralNorm(self.theta)
            self.phi = SpectralNorm(self.phi)
            self.g = SpectralNorm(self.g)
            self.o = SpectralNorm(self.o)
        self.gamma = nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, x, y=None):
        theta = self.theta(x)
        phi = F.max_pool2d(self.phi(x), [2, 2])
        g = F.max_pool2d(self.g(x), [2, 2])
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)
        beta = F.softmax(torch.bmm(theta.transpose(1, 2), phi), -1)
        o = self.o(
            torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2,
                                                    x.shape[2], x.shape[3]))
        return self.gamma * o + x


class MLAttrEncoder(nn.Module):

    def __init__(self):
        super(MLAttrEncoder, self).__init__()
        self.conv1 = Conv4x4(3, 32)
        self.conv2 = Conv4x4(32, 64)
        self.conv3 = Conv4x4(64, 128)
        self.conv4 = Conv4x4(128, 256)
        self.conv5 = Conv4x4(256, 512)
        self.conv6 = Conv4x4(512, 1024)
        self.conv7 = Conv4x4(1024, 1024)

        self.deconv1 = DeConv4x4(1024, 1024)
        self.deconv2 = DeConv4x4(2048, 512)
        self.deconv3 = DeConv4x4(1024, 256)
        self.deconv4 = DeConv4x4(512, 128)
        self.deconv5 = DeConv4x4(256, 64)
        self.deconv6 = DeConv4x4(128, 32)

        self.apply(init_func)

    def forward(self, Xt):
        feat1 = self.conv1(Xt)
        feat2 = self.conv2(feat1)
        feat3 = self.conv3(feat2)
        feat4 = self.conv4(feat3)
        feat5 = self.conv5(feat4)
        feat6 = self.conv6(feat5)
        z_attr1 = self.conv7(feat6)

        z_attr2 = self.deconv1(z_attr1, feat6)
        z_attr3 = self.deconv2(z_attr2, feat5)
        z_attr4 = self.deconv3(z_attr3, feat4)
        z_attr5 = self.deconv4(z_attr4, feat3)
        z_attr6 = self.deconv5(z_attr5, feat2)
        z_attr7 = self.deconv6(z_attr6, feat1)
        z_attr8 = F.interpolate(
            z_attr7, scale_factor=2, mode='bilinear', align_corners=True)

        return z_attr1, z_attr2, z_attr3, z_attr4, z_attr5, z_attr6, z_attr7, z_attr8


class AADGenerator(nn.Module):

    def __init__(self, c_id=256):
        super(AADGenerator, self).__init__()
        self.up1 = nn.ConvTranspose2d(
            c_id, 1024, kernel_size=2, stride=1, padding=0)
        self.AADBlk1 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk2 = AAD_ResBlk(1024, 1024, 2048, c_id)
        self.AADBlk3 = AAD_ResBlk(1024, 1024, 1024, c_id)
        self.AADBlk4 = AAD_ResBlk(1024, 512, 512, c_id)
        self.AADBlk5 = AAD_ResBlk(512, 256, 256, c_id)
        self.AADBlk6 = AAD_ResBlk(256, 128, 128, c_id)
        self.AADBlk7 = AAD_ResBlk(128, 64, 64, c_id)
        self.AADBlk8 = AAD_ResBlk(64, 3, 64, c_id)

        self.sa = Attention(512, use_sn=True)

        self.apply(init_func)

    def forward(self, z_attr, z_id, deformation):

        m = self.up1(z_id.reshape(z_id.shape[0], -1, 1, 1))
        m2 = F.interpolate(
            self.AADBlk1(m, z_attr[0], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m3 = F.interpolate(
            self.AADBlk2(m2, z_attr[1], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m4 = F.interpolate(
            self.AADBlk3(m3, z_attr[2], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m5 = F.interpolate(
            self.AADBlk4(m4, z_attr[3], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m5 = self.sa(m5)
        m6 = F.interpolate(
            self.AADBlk5(m5, z_attr[4], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m7 = F.interpolate(
            self.AADBlk6(m6, z_attr[5], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)
        m8 = F.interpolate(
            self.AADBlk7(m7, z_attr[6], z_id),
            scale_factor=2,
            mode='bilinear',
            align_corners=True)

        y = self.AADBlk8(m8, z_attr[7], z_id)

        return torch.tanh(y)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(
                deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation), deformation


class AEI_Net(nn.Module):

    def __init__(self, c_id=256, num_kp=17, device=torch.device('cuda')):
        super(AEI_Net, self).__init__()
        self.device = device
        self.encoder = MLAttrEncoder()
        self.generator = AADGenerator(c_id)
        self.dense_motion_network = DenseMotionNetwork(
            num_kp=num_kp, num_channels=3, estimate_occlusion_map=False)

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(
                deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return F.grid_sample(inp, deformation), deformation

    def flow_change(self, x, flow):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1
        grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)),
                         -1).unsqueeze(0).to(self.device)
        flow_delta = flow - grid
        return flow_delta

    def forward(self, Xt, z_id, kp_fuse, kp_t):
        output_flow = {}
        dense_motion = self.dense_motion_network(
            source_image=Xt, kp_driving=kp_fuse, kp_source=kp_t)
        deformation = dense_motion['deformation']

        with torch.no_grad():
            Xt_warp, _ = self.deform_input(Xt, deformation)
        attr = self.encoder(Xt_warp)

        Y = self.generator(attr, z_id, deformation)

        output_flow['deformed'], flow = self.deform_input(Xt, deformation)
        output_flow['flow'] = self.flow_change(Xt, flow)

        return Y, attr, output_flow

    def get_attr(self, X):
        return self.encoder(X)
