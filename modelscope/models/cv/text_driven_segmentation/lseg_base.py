# Adapted from https://github.com/isl-org/lang-seg.
# Originally MIT License, Copyright (c) 2021 Intelligent Systems Lab Org.

import torch
import torch.nn as nn

from .lseg_net import LSeg


class TextDrivenSegmentation(nn.Module):

    def __init__(self, model_dir):
        super(TextDrivenSegmentation, self).__init__()
        self.net = LSeg(model_dir=model_dir)
        self.model_dir = model_dir

    def forward(self, img, txt_list):
        b = img.size()[0]
        batch_name_list = txt_list
        xout_list = []
        for i in range(b):
            labelset = ['others', batch_name_list[i]]
            xout = self.net(img[i:i + 1], labelset=labelset)
            xout_list.append(xout)
        score_map = torch.cat(xout_list, dim=0)
        return score_map
