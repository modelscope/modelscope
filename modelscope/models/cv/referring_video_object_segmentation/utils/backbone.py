# The implementation is adopted from MTTR,
# made publicly available under the Apache 2.0 License at https://github.com/mttr2021/MTTR

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from .misc import NestedTensor, is_main_process
from .swin_transformer import SwinTransformer3D


class VideoSwinTransformerBackbone(nn.Module):
    """
    A wrapper which allows using Video-Swin Transformer as a temporal encoder for MTTR.
    Check out video-swin's original paper at: https://arxiv.org/abs/2106.13230 for more info about this architecture.
    Only the 'tiny' version of video swin was tested and is currently supported in our project.
    Additionally, we slightly modify video-swin to make it output per-frame embeddings as required by MTTR (check our
    paper's supplementary for more details), and completely discard of its 4th block.
    """

    def __init__(self, backbone_pretrained, backbone_pretrained_path,
                 train_backbone, running_mode, **kwargs):
        super(VideoSwinTransformerBackbone, self).__init__()
        # patch_size is (1, 4, 4) instead of the original (2, 4, 4).
        # this prevents swinT's original temporal downsampling so we can get per-frame features.
        swin_backbone = SwinTransformer3D(
            patch_size=(1, 4, 4),
            embed_dim=96,
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_size=(8, 7, 7),
            drop_path_rate=0.1,
            patch_norm=True)
        if backbone_pretrained and running_mode == 'train':
            state_dict = torch.load(backbone_pretrained_path)['state_dict']
            # extract swinT's kinetics-400 pretrained weights and ignore the rest (prediction head etc.)
            state_dict = {
                k[9:]: v
                for k, v in state_dict.items() if 'backbone.' in k
            }

            # sum over the patch embedding weight temporal dim  [96, 3, 2, 4, 4] --> [96, 3, 1, 4, 4]
            patch_embed_weight = state_dict['patch_embed.proj.weight']
            patch_embed_weight = patch_embed_weight.sum(dim=2, keepdims=True)
            state_dict['patch_embed.proj.weight'] = patch_embed_weight
            swin_backbone.load_state_dict(state_dict)

        self.patch_embed = swin_backbone.patch_embed
        self.pos_drop = swin_backbone.pos_drop
        self.layers = swin_backbone.layers[:-1]
        self.downsamples = nn.ModuleList()
        for layer in self.layers:
            self.downsamples.append(layer.downsample)
            layer.downsample = None
        self.downsamples[
            -1] = None  # downsampling after the last layer is not necessary

        self.layer_output_channels = [
            swin_backbone.embed_dim * 2**i for i in range(len(self.layers))
        ]
        self.train_backbone = train_backbone
        if not train_backbone:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, samples: NestedTensor):
        vid_frames = rearrange(samples.tensors, 't b c h w -> b c t h w')

        vid_embeds = self.patch_embed(vid_frames)
        vid_embeds = self.pos_drop(vid_embeds)
        layer_outputs = []  # layer outputs before downsampling
        for layer, downsample in zip(self.layers, self.downsamples):
            vid_embeds = layer(vid_embeds.contiguous())
            layer_outputs.append(vid_embeds)
            if downsample:
                vid_embeds = rearrange(vid_embeds, 'b c t h w -> b t h w c')
                vid_embeds = downsample(vid_embeds)
                vid_embeds = rearrange(vid_embeds, 'b t h w c -> b c t h w')
        layer_outputs = [
            rearrange(o, 'b c t h w -> t b c h w') for o in layer_outputs
        ]

        outputs = []
        orig_pad_mask = samples.mask
        for l_out in layer_outputs:
            pad_mask = F.interpolate(
                orig_pad_mask.float(), size=l_out.shape[-2:]).to(torch.bool)
            outputs.append(NestedTensor(l_out, pad_mask))
        return outputs

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FrozenBatchNorm2d(torch.nn.Module):
    """
    Modified from DETR https://github.com/facebookresearch/detr
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer('weight', torch.ones(n))
        self.register_buffer('bias', torch.zeros(n))
        self.register_buffer('running_mean', torch.zeros(n))
        self.register_buffer('running_var', torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class ResNetBackbone(nn.Module):
    """
    Modified from DETR https://github.com/facebookresearch/detr
    ResNet backbone with frozen BatchNorm.
    """

    def __init__(self,
                 backbone_name: str = 'resnet50',
                 train_backbone: bool = True,
                 dilation: bool = True,
                 **kwargs):
        super(ResNetBackbone, self).__init__()
        backbone = getattr(torchvision.models, backbone_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(),
            norm_layer=FrozenBatchNorm2d)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        return_layers = {
            'layer1': '0',
            'layer2': '1',
            'layer3': '2',
            'layer4': '3'
        }
        self.body = IntermediateLayerGetter(
            backbone, return_layers=return_layers)
        output_channels = 512 if backbone_name in ('resnet18',
                                                   'resnet34') else 2048
        self.layer_output_channels = [
            output_channels // 8, output_channels // 4, output_channels // 2,
            output_channels
        ]

    def forward(self, tensor_list: NestedTensor):
        t, b, _, _, _ = tensor_list.tensors.shape
        video_frames = rearrange(tensor_list.tensors,
                                 't b c h w -> (t b) c h w')
        padding_masks = rearrange(tensor_list.mask, 't b h w -> (t b) h w')
        features_list = self.body(video_frames)
        out = []
        for _, f in features_list.items():
            resized_padding_masks = F.interpolate(
                padding_masks[None].float(),
                size=f.shape[-2:]).to(torch.bool)[0]
            f = rearrange(f, '(t b) c h w -> t b c h w', t=t, b=b)
            resized_padding_masks = rearrange(
                resized_padding_masks, '(t b) h w -> t b h w', t=t, b=b)
            out.append(NestedTensor(f, resized_padding_masks))
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def init_backbone(backbone_name, **kwargs):
    if backbone_name == 'swin-t':
        return VideoSwinTransformerBackbone(**kwargs)
    elif 'resnet' in backbone_name:
        return ResNetBackbone(backbone_name, **kwargs)
    assert False, f'error: backbone "{backbone_name}" is not supported'
