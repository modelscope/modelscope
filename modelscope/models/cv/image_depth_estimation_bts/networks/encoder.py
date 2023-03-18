# The implementation is modified from ErenBalatkan/Bts-PyTorch
# made publicly available under the MIT license
# https://github.com/ErenBalatkan/Bts-PyTorch/blob/master/BTS.py

import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):

    def __init__(self, pretrained=False):
        super(Encoder, self).__init__()
        self.dense_op_h2 = None
        self.dense_op_h4 = None
        self.dense_op_h8 = None
        self.dense_op_h16 = None
        self.dense_features = None

        self.dense_feature_extractor = self.initial_feature_extractor(
            pretrained)
        self.freeze_batch_norm()
        self.initialize_hooks()

    def freeze_batch_norm(self):
        for module in self.dense_feature_extractor.modules():
            if isinstance(module, nn.modules.BatchNorm2d):
                module.track_running_stats = True
                module.eval()
                module.affine = True
                module.requires_grad = True

    def initial_feature_extractor(self, pretrained=False):
        dfe = models.densenet161(pretrained=pretrained)
        dfe.features.denseblock1.requires_grad = False
        dfe.features.denseblock2.requires_grad = False
        dfe.features.conv0.requires_grad = False
        return dfe

    def set_h2(self, module, input_, output):
        self.dense_op_h2 = output

    def set_h4(self, module, input_, output):
        self.dense_op_h4 = output

    def set_h8(self, module, input_, output):
        self.dense_op_h8 = output

    def set_h16(self, module, input_, output):
        self.dense_op_h16 = output

    def set_dense_features(self, module, input_, output):
        self.dense_features = output

    def initialize_hooks(self):
        self.dense_feature_extractor.features.relu0.register_forward_hook(
            self.set_h2)
        self.dense_feature_extractor.features.pool0.register_forward_hook(
            self.set_h4)
        self.dense_feature_extractor.features.transition1.register_forward_hook(
            self.set_h8)
        self.dense_feature_extractor.features.transition2.register_forward_hook(
            self.set_h16)
        self.dense_feature_extractor.features.norm5.register_forward_hook(
            self.set_dense_features)

    def forward(self, x):
        _ = self.dense_feature_extractor(x)
        joint_input = (self.dense_features.relu(), self.dense_op_h2,
                       self.dense_op_h4, self.dense_op_h8, self.dense_op_h16)
        return joint_input
