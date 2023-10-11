# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
"""Superresolution network architectures from the paper
"Efficient Geometry-aware 3D Generative Adversarial Networks"."""

import numpy as np
import torch

from modelscope.ops.image_control_3d_portrait.torch_utils import (misc,
                                                                  persistence)
from modelscope.ops.image_control_3d_portrait.torch_utils.ops import upfirdn2d
from .networks_stylegan2 import (Conv2dLayer, SynthesisBlock, SynthesisLayer,
                                 ToRGBLayer)


# for 512x512 generation
@persistence.persistent_class
class SuperresolutionHybrid8X(torch.nn.Module):

    def __init__(
            self,
            channels,
            img_resolution,
            sr_num_fp16_res,
            sr_antialias,
            num_fp16_res=4,
            conv_clamp=None,
            channel_base=None,
            channel_max=None,  # IGNORE
            **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlock(
            channels,
            128,
            w_dim=512,
            resolution=256,
            img_channels=3,
            is_last=False,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.block1 = SynthesisBlock(
            128,
            64,
            w_dim=512,
            resolution=512,
            img_channels=3,
            is_last=True,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


# for 256x256 generation
@persistence.persistent_class
class SuperresolutionHybrid4X(torch.nn.Module):

    def __init__(
            self,
            channels,
            img_resolution,
            sr_num_fp16_res,
            sr_antialias,
            num_fp16_res=4,
            conv_clamp=None,
            channel_base=None,
            channel_max=None,  # IGNORE
            **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0
        self.sr_antialias = sr_antialias
        self.input_resolution = 128
        self.block0 = SynthesisBlockNoUp(
            channels,
            128,
            w_dim=512,
            resolution=128,
            img_channels=3,
            is_last=False,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.block1 = SynthesisBlock(
            128,
            64,
            w_dim=512,
            resolution=256,
            img_channels=3,
            is_last=True,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] < self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


# for 128 x 128 generation
@persistence.persistent_class
class SuperresolutionHybrid2X(torch.nn.Module):

    def __init__(
            self,
            channels,
            img_resolution,
            sr_num_fp16_res,
            sr_antialias,
            num_fp16_res=4,
            conv_clamp=None,
            channel_base=None,
            channel_max=None,  # IGNORE
            **block_kwargs):
        super().__init__()
        assert img_resolution == 128

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 64
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlockNoUp(
            channels,
            128,
            w_dim=512,
            resolution=64,
            img_channels=3,
            is_last=False,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.block1 = SynthesisBlock(
            128,
            64,
            w_dim=512,
            resolution=128,
            img_channels=3,
            is_last=True,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


# TODO: Delete (here for backwards compatibility with old 256x256 models)
@persistence.persistent_class
class SuperresolutionHybridDeepfp32(torch.nn.Module):

    def __init__(
            self,
            channels,
            img_resolution,
            sr_num_fp16_res,
            num_fp16_res=4,
            conv_clamp=None,
            channel_base=None,
            channel_max=None,  # IGNORE
            **block_kwargs):
        super().__init__()
        assert img_resolution == 256
        use_fp16 = sr_num_fp16_res > 0

        self.input_resolution = 128
        self.block0 = SynthesisBlockNoUp(
            channels,
            128,
            w_dim=512,
            resolution=128,
            img_channels=3,
            is_last=False,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.block1 = SynthesisBlock(
            128,
            64,
            w_dim=512,
            resolution=256,
            img_channels=3,
            is_last=True,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter([1, 3, 3, 1]))

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] < self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False)
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb


@persistence.persistent_class
class SynthesisBlockNoUp(torch.nn.Module):

    def __init__(
            self,
            in_channels,  # Number of input channels, 0 = first block.
            out_channels,  # Number of output channels.
            w_dim,  # Intermediate latent (W) dimensionality.
            resolution,  # Resolution of this block.
            img_channels,  # Number of output color channels.
            is_last,  # Is this the last block?
            architecture='skip',  # Architecture: 'orig', 'skip', 'resnet'.
            resample_filter=[
                1, 3, 3, 1
            ],  # Low-pass filter to apply when resampling activations.
            conv_clamp=256,  # Clamp the output of convolution layers to +-X, None = disable clamping.
            use_fp16=False,  # Use FP16 for this block?
            fp16_channels_last=False,  # Use channels-last memory format with FP16?
            fused_modconv_default=True,  # Default value of fused_modconv.
            **layer_kwargs,  # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.fused_modconv_default = fused_modconv_default
        self.register_buffer('resample_filter',
                             upfirdn2d.setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = torch.nn.Parameter(
                torch.randn([out_channels, resolution, resolution]))

        if in_channels != 0:
            self.conv0 = SynthesisLayer(
                in_channels,
                out_channels,
                w_dim=w_dim,
                resolution=resolution,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last,
                **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(
            out_channels,
            out_channels,
            w_dim=w_dim,
            resolution=resolution,
            conv_clamp=conv_clamp,
            channels_last=self.channels_last,
            **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(
                out_channels,
                img_channels,
                w_dim=w_dim,
                conv_clamp=conv_clamp,
                channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                up=2,
                resample_filter=resample_filter,
                channels_last=self.channels_last)

    def forward(self,
                x,
                img,
                ws,
                force_fp32=False,
                fused_modconv=None,
                update_emas=False,
                **layer_kwargs):
        _ = update_emas  # unused
        misc.assert_shape(ws,
                          [None, self.num_conv + self.num_torgb, self.w_dim])
        w_iter = iter(ws.unbind(dim=1))
        if ws.device.type != 'cuda':
            force_fp32 = True
        dtype = torch.float16 if self.use_fp16 and not force_fp32 else torch.float32
        memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = self.fused_modconv_default
        if fused_modconv == 'inference_only':
            fused_modconv = (not self.training)

        # Input.
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
        else:
            misc.assert_shape(
                x, [None, self.in_channels, self.resolution, self.resolution])
            x = x.to(dtype=dtype, memory_format=memory_format)

        # Main layers.
        if self.in_channels == 0:
            x = self.conv1(
                x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            x = self.conv0(
                x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(
                x,
                next(w_iter),
                fused_modconv=fused_modconv,
                gain=np.sqrt(0.5),
                **layer_kwargs)
            x = y.add_(x)
        else:
            x = self.conv0(
                x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(
                x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

        # ToRGB.
        # if img is not None:
        # misc.assert_shape(img, [None, self.img_channels, self.resolution // 2, self.resolution // 2])
        # img = upfirdn2d.upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(
                dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == torch.float32
        return x, img

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}'


# for 512x512 generation
@persistence.persistent_class
class SuperresolutionHybrid8XDC(torch.nn.Module):

    def __init__(
            self,
            channels,
            img_resolution,
            sr_num_fp16_res,
            sr_antialias,
            num_fp16_res=4,
            conv_clamp=None,
            channel_base=None,
            channel_max=None,  # IGNORE
            **block_kwargs):
        super().__init__()
        assert img_resolution == 512

        use_fp16 = sr_num_fp16_res > 0
        self.input_resolution = 128
        self.sr_antialias = sr_antialias
        self.block0 = SynthesisBlock(
            channels,
            256,
            w_dim=512,
            resolution=256,
            img_channels=3,
            is_last=False,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)
        self.block1 = SynthesisBlock(
            256,
            128,
            w_dim=512,
            resolution=512,
            img_channels=3,
            is_last=True,
            use_fp16=use_fp16,
            conv_clamp=(256 if use_fp16 else None),
            **block_kwargs)

    def forward(self, rgb, x, ws, **block_kwargs):
        ws = ws[:, -1:, :].repeat(1, 3, 1)

        if x.shape[-1] != self.input_resolution:
            x = torch.nn.functional.interpolate(
                x,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(self.input_resolution, self.input_resolution),
                mode='bilinear',
                align_corners=False,
                antialias=self.sr_antialias)

        x, rgb = self.block0(x, rgb, ws, **block_kwargs)
        x, rgb = self.block1(x, rgb, ws, **block_kwargs)
        return rgb
