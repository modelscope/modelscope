# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from .unet import UNet


class DeRetouchingModule():

    def __init__(self, model_path):
        self.retouching_network = UNet(3, 3).to('cuda')
        self.retouching_network.load_state_dict(
            torch.load(model_path, map_location='cpu')['generator'])
        self.retouching_network.eval()

    def run(self, face_albedo_map, texture_map):
        """

        :param face_albedo_map: tensor, (1, 3, 256, 256), 0~1, rgb
        :param texture_map: tensor, (1, 3, 256, 256), -1~1, rgb
        :return:
        """
        h, w = texture_map.shape[2:]
        retouch_input = torch.nn.functional.interpolate(
            texture_map, (512, 512), mode='bilinear')

        # predict blend layer
        blend_layer = self.retouching_network(retouch_input)  # value: 0~1
        blend_layer = torch.nn.functional.interpolate(
            blend_layer, (h, w), mode='bilinear')

        # retouch texture map
        tex = (texture_map + 1.0) / 2
        retouched_tex = (1 - 2 * blend_layer
                         ) * tex * tex + 2 * blend_layer * tex  # value: 0~1

        # de-retouch albedo map
        A_0 = face_albedo_map
        T_0 = retouched_tex
        T_ = tex

        phi = 1e6 * T_0 * T_0 * T_0 + 1e-6
        B = (T_ + 1 / phi) / (T_0 + 1 / phi)
        A_ = A_0 * B

        de_retouched_albedo = A_

        return de_retouched_albedo
