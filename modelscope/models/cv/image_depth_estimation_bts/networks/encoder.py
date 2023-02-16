# The implementation is modified from cleinc / bts
# made publicly available under the GPL-3.0-or-later
# https://github.com/cleinc/bts/blob/master/pytorch/bts.py

import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self, encoder='densenet161_bts', pretrained=False):
        super(Encoder, self).__init__()
        self.encoder = encoder

        if encoder == 'densenet121_bts':
            self.base_model = models.densenet121(
                pretrained=pretrained).features
            self.feat_names = [
                'relu0', 'pool0', 'transition1', 'transition2', 'norm5'
            ]
            self.feat_out_channels = [64, 64, 128, 256, 1024]
        elif encoder == 'densenet161_bts':
            self.base_model = models.densenet161(
                pretrained=pretrained).features
            self.feat_names = [
                'relu0', 'pool0', 'transition1', 'transition2', 'norm5'
            ]
            self.feat_out_channels = [96, 96, 192, 384, 2208]
        elif encoder == 'resnet50_bts':
            self.base_model = models.resnet50(pretrained=pretrained)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        elif encoder == 'resnet101_bts':
            self.base_model = models.resnet101(pretrained=pretrained)
            self.feat_names = ['relu', 'layer1', 'layer2', 'layer3', 'layer4']
            self.feat_out_channels = [64, 256, 512, 1024, 2048]
        else:
            raise NotImplementedError

    def forward(self, x):
        features = [x]
        skip_feat = [x]
        for k, v in self.base_model._modules.items():
            if 'resnet' in self.encoder and ('fc' in k or 'avgpool' in k):
                continue
            feature = v(features[-1])
            features.append(feature)
            if any(x in k for x in self.feat_names):
                skip_feat.append(feature)

        return skip_feat
