# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from modelscope.models.cv.video_frame_interpolation.flow_model.raft import RAFT
from modelscope.models.cv.video_frame_interpolation.interp_model.IFNet_swin import \
    IFNet
from modelscope.models.cv.video_frame_interpolation.interp_model.refinenet_arch import (
    InterpNet, InterpNetDs)


class VFINet(nn.Module):

    def __init__(self, args, Ds_flag=False):
        super(VFINet, self).__init__()
        self.flownet = RAFT(args)
        self.internet = InterpNet()
        if Ds_flag:
            self.internet_Ds = InterpNetDs()

    def img_trans(self, img_tensor):  # in format of RGB
        img_tensor = img_tensor / 255.0
        mean = torch.Tensor([0.429, 0.431, 0.397]).view(1, 3, 1,
                                                        1).type_as(img_tensor)
        img_tensor -= mean
        return img_tensor

    def add_mean(self, x):
        mean = torch.Tensor([0.429, 0.431, 0.397]).view(1, 3, 1, 1).type_as(x)
        return x + mean

    def forward(self, imgs, timestep=0.5):
        self.flownet.eval()
        self.internet.eval()
        with torch.no_grad():
            img0 = imgs[:, :3]
            img1 = imgs[:, 3:6]
            img2 = imgs[:, 6:9]
            img3 = imgs[:, 9:12]

            _, F10_up = self.flownet(img1, img0, iters=12, test_mode=True)
            _, F12_up = self.flownet(img1, img2, iters=12, test_mode=True)
            _, F21_up = self.flownet(img2, img1, iters=12, test_mode=True)
            _, F23_up = self.flownet(img2, img3, iters=12, test_mode=True)

            img1 = self.img_trans(img1.clone())
            img2 = self.img_trans(img2.clone())

            It_warp = self.internet(
                img1, img2, F10_up, F12_up, F21_up, F23_up, timestep=timestep)
            It_warp = self.add_mean(It_warp)

        return It_warp
