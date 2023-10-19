# https://github.com/eladrich/pixel2style2pixel
import torch
from torch import nn

from .model_irse import Backbone


class IDFeatures(nn.Module):

    def __init__(self, model_path):
        super(IDFeatures, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(
            input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def forward(self, x, crop=False):
        # Not sure of the image range here
        if crop:
            x = torch.nn.functional.interpolate(x, (256, 256), mode='area')
            x = x[:, :, 35:223, 32:220]
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
