# Adopted from https://github.com/Limingxing00/RDE-VOS-CVPR2022
# under MIT License

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.cv.video_object_segmentation.modules import (
    KeyEncoder, KeyProjection, MemCrompress, ResBlock, UpsampleBlock,
    ValueEncoder, ValueEncoderSO)


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.compress = ResBlock(1024, 512)
        self.up_16_8 = UpsampleBlock(512, 512, 256)  # 1/16 -> 1/8
        self.up_8_4 = UpsampleBlock(256, 256, 256)  # 1/8 -> 1/4

        self.pred = nn.Conv2d(
            256, 1, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, f16, f8, f4):
        x = self.compress(f16)
        x = self.up_16_8(f8, x)
        x = self.up_8_4(f4, x)

        x = self.pred(F.relu(x))

        x = F.interpolate(
            x, scale_factor=4, mode='bilinear', align_corners=False)
        return x


class MemoryReader(nn.Module):

    def __init__(self):
        super().__init__()

    def get_affinity(self, mk, qk):
        B, CK, T, H, W = mk.shape
        mk = mk.flatten(start_dim=2)
        qk = qk.flatten(start_dim=2)

        a = mk.pow(2).sum(1).unsqueeze(2)
        b = 2 * (mk.transpose(1, 2) @ qk)
        # this term will be cancelled out in the softmax
        # c = qk.pow(2).sum(1).unsqueeze(1)

        affinity = (-a + b) / math.sqrt(CK)  # B, THW, HW

        # softmax operation; aligned the evaluation style
        maxes = torch.max(affinity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(affinity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum

        return affinity

    def readout(self, affinity, mv, qv):
        B, CV, T, H, W = mv.shape

        mo = mv.view(B, CV, T * H * W)
        mem = torch.bmm(mo, affinity)  # Weighted-sum B, CV, HW
        mem = mem.view(B, CV, H, W)

        mem_out = torch.cat([mem, qv], dim=1)

        return mem_out


class RDE_VOS(nn.Module):

    def __init__(self, single_object, repeat=0, norm=False):
        super().__init__()
        self.single_object = single_object

        self.key_encoder = KeyEncoder()
        if single_object:
            self.value_encoder = ValueEncoderSO()
        else:
            self.value_encoder = ValueEncoder()
            # Compress memory bank
            self.mem_compress = MemCrompress(repeat=repeat, norm=norm)
            # self.mem_compress.train(True)

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in decoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.memory = MemoryReader()
        self.decoder = Decoder()

    def aggregate(self, prob):
        # During training, torch.prod work on channel dimension.
        new_prob = torch.cat([torch.prod(1 - prob, dim=1, keepdim=True), prob],
                             1).clamp(1e-7, 1 - 1e-7)
        logits = torch.log((new_prob / (1 - new_prob)))
        return logits

    def encode_key(self, frame):
        # input: b*t*c*h*w
        b, t = frame.shape[:2]

        f16, f8, f4 = self.key_encoder(frame.flatten(start_dim=0, end_dim=1))
        k16 = self.key_proj(f16)
        f16_thin = self.key_comp(f16)

        # B*C*T*H*W
        k16 = k16.view(b, t, *k16.shape[-3:]).transpose(1, 2).contiguous()

        # B*T*C*H*W
        f16_thin = f16_thin.view(b, t, *f16_thin.shape[-3:])
        f16 = f16.view(b, t, *f16.shape[-3:])
        f8 = f8.view(b, t, *f8.shape[-3:])
        f4 = f4.view(b, t, *f4.shape[-3:])

        return k16, f16_thin, f16, f8, f4

    def encode_value(self, frame, kf16, mask, other_mask=None):
        # Extract memory key/value for a frame
        if self.single_object:
            f16 = self.value_encoder(frame, kf16, mask)
        else:
            f16 = self.value_encoder(frame, kf16, mask, other_mask)
        return f16.unsqueeze(2)  # B*512*T*H*W

    def segment(self, qk16, qv16, qf8, qf4, mk16, mv16, selector=None):
        # q - query, m - memory
        # qv16 is f16_thin above
        affinity = self.memory.get_affinity(mk16, qk16)

        if self.single_object:
            mv2qv = self.memory.readout(affinity, mv16, qv16)
            logits = self.decoder(mv2qv, qf8, qf4)
            prob = torch.sigmoid(logits)
        else:
            mv2qv_o1 = self.memory.readout(affinity, mv16[:, 0], qv16)
            mv2qv_o2 = self.memory.readout(affinity, mv16[:, 1], qv16)
            logits = torch.cat([
                self.decoder(mv2qv_o1, qf8, qf4),
                self.decoder(mv2qv_o2, qf8, qf4),
            ], 1)

            prob = torch.sigmoid(logits)
            prob = prob * selector.unsqueeze(2).unsqueeze(2)

        logits = self.aggregate(prob)
        prob = F.softmax(logits, dim=1)[:, 1:]

        if self.single_object:
            return logits, prob, mv2qv
        else:
            return logits, prob, mv2qv_o1, mv2qv_o2

    def memCrompress(self, key, value):
        return self.mem_compress(key, value)

    def forward(self, mode, *args, **kwargs):
        if mode == 'encode_key':
            return self.encode_key(*args, **kwargs)
        elif mode == 'encode_value':
            return self.encode_value(*args, **kwargs)
        elif mode == 'segment':
            return self.segment(*args, **kwargs)
        elif mode == 'compress':
            return self.memCrompress(*args, **kwargs)
        else:
            raise NotImplementedError
