import collections
import os
import sys

import torch
from torch import nn

from .base_module import *

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))


class Generator(nn.Module):

    def __init__(
        self,
        size,
        semantic_dim,
        channels,
        num_labels,
        match_kernels,
        blur_kernel=[1, 3, 3, 1],
        wavelet_down_levels={'16': 3},
        window_size=8,
    ):
        super().__init__()
        self.size = size
        self.reference_encoder = Encoder_wiflow(size, 3, channels, num_labels,
                                                match_kernels, blur_kernel)

        self.skeleton_encoder = Encoder_wiflow(
            size,
            semantic_dim,
            channels,
        )

        self.target_image_renderer = Decoder_wiflow_wavelet_fuse25(
            size, channels, num_labels, match_kernels, blur_kernel,
            wavelet_down_levels, window_size)

    def _cal_temp(self, module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def forward(self, source_image, skeleton, kp_skeleton):
        output_dict = {}
        recoder = collections.defaultdict(list)
        skeleton_feature_list, source_feature_list = [], []
        skeleton_feature = self.skeleton_encoder(
            skeleton, out_list=skeleton_feature_list)
        _ = self.reference_encoder(
            source_image, recoder, out_list=source_feature_list)
        neural_textures = recoder['neural_textures']

        output_dict['fake_image'], delta_x_all, delta_y_all, delta_list, last_flow_all, mask_all_h, mask_all_l = \
            self.target_image_renderer(skeleton_feature, neural_textures, skeleton_feature_list,
                                       source_feature_list, kp_skeleton, recoder)
        output_dict['info'] = recoder
        output_dict['delta_x'] = delta_x_all
        output_dict['delta_y'] = delta_y_all
        output_dict['delta_list'] = delta_list
        output_dict['last_flow_all'] = last_flow_all
        output_dict['mask_all_h'] = mask_all_h
        output_dict['mask_all_l'] = mask_all_l
        return output_dict
