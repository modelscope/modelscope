""" VideoInpaintingProcess
The implementation here is modified based on STTN,
 originally Apache 2.0 License and publicly available at https://github.com/researchmm/STTN
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from modelscope.metainfo import Models
from modelscope.models import Model
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


class BaseNetwork(nn.Module):

    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(
            'Network [%s] was created. Total number of parameters: %.1f million. '
            'To see the architecture, do print(network).' %
            (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1
                                           or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented'
                        % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


@MODELS.register_module(
    Tasks.video_inpainting, module_name=Models.video_inpainting)
class VideoInpainting(TorchModel):

    def __init__(self, model_dir, device_id=0, *args, **kwargs):
        super().__init__(
            model_dir=model_dir, device_id=device_id, *args, **kwargs)
        self.model = InpaintGenerator()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        pretrained_params = torch.load(
            '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            map_location=device)
        self.model.load_state_dict(pretrained_params['netG'])
        self.model.eval()
        self.device_id = device_id
        if self.device_id >= 0 and torch.cuda.is_available():
            self.model.to('cuda:{}'.format(self.device_id))
            logger.info('Use GPU: {}'.format(self.device_id))
        else:
            self.device_id = -1
            logger.info('Use CPU for inference')


class InpaintGenerator(BaseNetwork):

    def __init__(self, init_weights=True):
        super(InpaintGenerator, self).__init__()
        channel = 256
        stack_num = 6
        patchsize = [(48, 24), (16, 8), (8, 4), (4, 2)]
        blocks = []
        for _ in range(stack_num):
            blocks.append(TransformerBlock(patchsize, hidden=channel))
        self.transformer = nn.Sequential(*blocks)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, channel, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            deconv(channel, 128, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            deconv(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1))

        if init_weights:
            self.init_weights()

    def forward(self, masked_frames, masks):
        b, t, c, h, w = masked_frames.size()
        masks = masks.view(b * t, 1, h, w)
        enc_feat = self.encoder(masked_frames.view(b * t, c, h, w))
        _, c, h, w = enc_feat.size()
        masks = F.interpolate(masks, scale_factor=1.0 / 4)
        enc_feat = self.transformer({
            'x': enc_feat,
            'm': masks,
            'b': b,
            'c': c
        })['x']
        output = self.decoder(enc_feat)
        output = torch.tanh(output)
        return output

    def infer(self, feat, masks):
        t, c, h, w = masks.size()
        masks = masks.view(t, c, h, w)
        masks = F.interpolate(masks, scale_factor=1.0 / 4)
        t, c, _, _ = feat.size()
        enc_feat = self.transformer({
            'x': feat,
            'm': masks,
            'b': 1,
            'c': c
        })['x']
        return enc_feat


class deconv(nn.Module):

    def __init__(self,
                 input_channel,
                 output_channel,
                 kernel_size=3,
                 padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            input_channel,
            output_channel,
            kernel_size=kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        x = F.interpolate(
            x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, m):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            query.size(-1))
        scores.masked_fill(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsize, d_model):
        super().__init__()
        self.patchsize = patchsize
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        self.attention = Attention()

    def forward(self, x, m, b, c):
        bt, _, h, w = x.size()
        t = bt // b
        d_k = c // len(self.patchsize)
        output = []
        _query = self.query_embedding(x)
        _key = self.key_embedding(x)
        _value = self.value_embedding(x)
        for (width, height), query, key, value in zip(
                self.patchsize,
                torch.chunk(_query, len(self.patchsize), dim=1),
                torch.chunk(_key, len(self.patchsize), dim=1),
                torch.chunk(_value, len(self.patchsize), dim=1)):
            out_w, out_h = w // width, h // height
            mm = m.view(b, t, 1, out_h, height, out_w, width)
            mm = mm.permute(0, 1, 3, 5, 2, 4,
                            6).contiguous().view(b, t * out_h * out_w,
                                                 height * width)
            mm = (mm.mean(-1) > 0.5).unsqueeze(1).repeat(
                1, t * out_h * out_w, 1)
            query = query.view(b, t, d_k, out_h, height, out_w, width)
            query = query.permute(0, 1, 3, 5, 2, 4,
                                  6).contiguous().view(b, t * out_h * out_w,
                                                       d_k * height * width)
            key = key.view(b, t, d_k, out_h, height, out_w, width)
            key = key.permute(0, 1, 3, 5, 2, 4,
                              6).contiguous().view(b, t * out_h * out_w,
                                                   d_k * height * width)
            value = value.view(b, t, d_k, out_h, height, out_w, width)
            value = value.permute(0, 1, 3, 5, 2, 4,
                                  6).contiguous().view(b, t * out_h * out_w,
                                                       d_k * height * width)
            y, _ = self.attention(query, key, value, mm)
            y = y.view(b, t, out_h, out_w, d_k, height, width)
            y = y.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(bt, d_k, h, w)
            output.append(y)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        return x


class FeedForward(nn.Module):

    def __init__(self, d_model):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, patchsize, hidden=128):  # hidden=128
        super().__init__()
        self.attention = MultiHeadedAttention(patchsize, d_model=hidden)
        self.feed_forward = FeedForward(hidden)

    def forward(self, x):
        x, m, b, c = x['x'], x['m'], x['b'], x['c']
        x = x + self.attention(x, m, b, c)
        x = x + self.feed_forward(x)
        return {'x': x, 'm': m, 'b': b, 'c': c}


class Discriminator(BaseNetwork):

    def __init__(self,
                 in_channels=3,
                 use_sigmoid=False,
                 use_spectral_norm=True,
                 init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid
        nf = 64

        self.conv = nn.Sequential(
            spectral_norm(
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=nf * 1,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=1,
                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 1,
                    nf * 2,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 2,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(
                nn.Conv3d(
                    nf * 4,
                    nf * 4,
                    kernel_size=(3, 5, 5),
                    stride=(1, 2, 2),
                    padding=(1, 2, 2),
                    bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(
                nf * 4,
                nf * 4,
                kernel_size=(3, 5, 5),
                stride=(1, 2, 2),
                padding=(1, 2, 2)))

        if init_weights:
            self.init_weights()

    def forward(self, xs):
        xs_t = torch.transpose(xs, 0, 1)
        xs_t = xs_t.unsqueeze(0)
        feat = self.conv(xs_t)
        if self.use_sigmoid:
            feat = torch.sigmoid(feat)
        out = torch.transpose(feat, 1, 2)
        return out


def spectral_norm(module, mode=True):
    if mode:
        return _spectral_norm(module)
    return module
