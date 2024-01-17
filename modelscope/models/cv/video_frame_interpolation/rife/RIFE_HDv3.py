# The implementation here is modified based on ECCV2022-RIFE,
# originally MIT License, Copyright  (c)  Megvii  Inc.,
# and publicly available at https://github.com/megvii-research/ECCV2022-RIFE

import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from modelscope.metainfo import Models
from modelscope.models.base import Tensor
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .IFNet_HDv3 import *
from .loss import *
from .warplayer import warp


@MODELS.register_module(
    Tasks.video_frame_interpolation, module_name=Models.rife)
class RIFEModel(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.flownet = IFNet()
        self.flownet.to(self.device)
        self.optimG = AdamW(
            self.flownet.parameters(), lr=1e-6, weight_decay=1e-4)
        self.epe = EPE()
        # self.vgg = VGGPerceptualLoss().to(device)
        self.sobel = SOBEL()
        self.load_model(model_dir, -1)
        self.eval()

    def train(self):
        self.flownet.train()

    def eval(self):
        self.flownet.eval()

    def load_model(self, path, rank=0):

        def convert(param):
            if rank == -1:
                return {
                    k.replace('module.', ''): v
                    for k, v in param.items() if 'module.' in k
                }
            else:
                return param

        if rank <= 0:
            if torch.cuda.is_available():
                self.flownet.load_state_dict(
                    convert(torch.load('{}/flownet.pkl'.format(path))))
            else:
                self.flownet.load_state_dict(
                    convert(
                        torch.load(
                            '{}/flownet.pkl'.format(path),
                            map_location='cpu')))

    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),
                       '{}/flownet.pkl'.format(path))

    def inference(self, img0, img1, scale=1.0):
        imgs = torch.cat((img0, img1), 1)
        scale_list = [4 / scale, 2 / scale, 1 / scale]
        _, _, merged = self.flownet(imgs, scale_list)
        return merged[2].detach()

    def forward(self, inputs):
        img0 = inputs['img0']
        img1 = inputs['img1']
        scale = inputs['scale']
        return {'output': self.inference(img0, img1, scale)}

    def update(self,
               imgs,
               gt,
               learning_rate=0,
               mul=1,
               training=True,
               flow_gt=None):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        # img0 = imgs[:, :3]
        # img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [4, 2, 1]
        flow, mask, merged = self.flownet(
            torch.cat((imgs, gt), 1), scale=scale, training=training)
        loss_l1 = (merged[2] - gt).abs().mean()
        loss_smooth = self.sobel(flow[2], flow[2] * 0).mean()
        # loss_vgg = self.vgg(merged[2], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = loss_cons + loss_smooth * 0.1
            loss_G.backward()
            self.optimG.step()
        # else:
        #     flow_teacher = flow[2]
        return merged[2], {
            'mask': mask,
            'flow': flow[2][:, :2],
            'loss_l1': loss_l1,
            'loss_cons': loss_cons,
            'loss_smooth': loss_smooth,
        }
