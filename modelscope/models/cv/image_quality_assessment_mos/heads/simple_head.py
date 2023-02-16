import torch
from torch import nn


class SimpleHead(torch.nn.Module):

    def __init__(self, feats_dims, out_num=1):
        super(SimpleHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        out_num = out_num
        self.fc_out = nn.Sequential(
            nn.Linear(feats_dims[-1], 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, out_num))

    def forward(self, x):
        x = x[-1]
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc_out(x)
        return x
