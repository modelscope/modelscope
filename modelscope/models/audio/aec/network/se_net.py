# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.models.audio.aec.layers.activations import (RectifiedLinear,
                                                            Sigmoid)
from modelscope.models.audio.aec.layers.affine_transform import AffineTransform
from modelscope.models.audio.aec.layers.deep_fsmn import DeepFsmn
from modelscope.models.audio.aec.layers.uni_deep_fsmn import (Conv2d,
                                                              UniDeepFsmn)


class MaskNet(nn.Module):

    def __init__(self,
                 indim,
                 outdim,
                 layers=9,
                 hidden_dim=128,
                 hidden_dim2=None,
                 lorder=20,
                 rorder=0,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 crm=False,
                 vad=False,
                 linearout=False):
        super(MaskNet, self).__init__()

        self.linear1 = AffineTransform(indim, hidden_dim)
        self.relu = RectifiedLinear(hidden_dim, hidden_dim)
        if hidden_dim2 is None:
            hidden_dim2 = hidden_dim

        if rorder == 0:
            repeats = [
                UniDeepFsmn(
                    hidden_dim,
                    hidden_dim,
                    lorder,
                    hidden_dim2,
                    dilation=dilation,
                    layer_norm=layer_norm,
                    dropout=dropout) for i in range(layers)
            ]
        else:
            repeats = [
                DeepFsmn(
                    hidden_dim,
                    hidden_dim,
                    lorder,
                    rorder,
                    hidden_dim2,
                    layer_norm=layer_norm,
                    dropout=dropout) for i in range(layers)
            ]
        self.deepfsmn = nn.Sequential(*repeats)

        self.linear2 = AffineTransform(hidden_dim, outdim)

        self.crm = crm
        if self.crm:
            self.sig = nn.Tanh()
        else:
            self.sig = Sigmoid(outdim, outdim)

        self.vad = vad
        if self.vad:
            self.linear3 = AffineTransform(hidden_dim, 1)

        self.layers = layers
        self.linearout = linearout
        if self.linearout and self.vad:
            print('Warning: not supported nnet')

    def forward(self, feat, ctl=None):
        x1 = self.linear1(feat)
        x2 = self.relu(x1)
        if ctl is not None:
            ctl = min(ctl, self.layers - 1)
            for i in range(ctl):
                x2 = self.deepfsmn[i](x2)
            mask = self.sig(self.linear2(x2))
            if self.vad:
                vad = torch.sigmoid(self.linear3(x2))
                return mask, vad
            else:
                return mask
        x3 = self.deepfsmn(x2)
        if self.linearout:
            return self.linear2(x3)
        mask = self.sig(self.linear2(x3))
        if self.vad:
            vad = torch.sigmoid(self.linear3(x3))
            return mask, vad
        else:
            return mask

    def to_kaldi_nnet(self):
        re_str = ''
        re_str += '<Nnet>\n'
        re_str += self.linear1.to_kaldi_nnet()
        re_str += self.relu.to_kaldi_nnet()
        for dfsmn in self.deepfsmn:
            re_str += dfsmn.to_kaldi_nnet()
        re_str += self.linear2.to_kaldi_nnet()
        re_str += self.sig.to_kaldi_nnet()
        re_str += '</Nnet>\n'

        return re_str

    def to_raw_nnet(self, fid):
        self.linear1.to_raw_nnet(fid)
        for dfsmn in self.deepfsmn:
            dfsmn.to_raw_nnet(fid)
        self.linear2.to_raw_nnet(fid)


class StageNet(nn.Module):

    def __init__(self,
                 indim,
                 outdim,
                 layers=9,
                 layers2=6,
                 hidden_dim=128,
                 lorder=20,
                 rorder=0,
                 layer_norm=False,
                 dropout=0,
                 crm=False,
                 vad=False,
                 linearout=False):
        super(StageNet, self).__init__()

        self.stage1 = nn.ModuleList()
        self.stage2 = nn.ModuleList()
        layer = nn.Sequential(nn.Linear(indim, hidden_dim), nn.ReLU())
        self.stage1.append(layer)
        for i in range(layers):
            layer = UniDeepFsmn(
                hidden_dim,
                hidden_dim,
                lorder,
                hidden_dim,
                layer_norm=layer_norm,
                dropout=dropout)
            self.stage1.append(layer)
        layer = nn.Sequential(nn.Linear(hidden_dim, 321), nn.Sigmoid())
        self.stage1.append(layer)
        # stage2
        layer = nn.Sequential(nn.Linear(321 + indim, hidden_dim), nn.ReLU())
        self.stage2.append(layer)
        for i in range(layers2):
            layer = UniDeepFsmn(
                hidden_dim,
                hidden_dim,
                lorder,
                hidden_dim,
                layer_norm=layer_norm,
                dropout=dropout)
            self.stage2.append(layer)
        layer = nn.Sequential(
            nn.Linear(hidden_dim, outdim),
            nn.Sigmoid() if not crm else nn.Tanh())
        self.stage2.append(layer)
        self.crm = crm
        self.vad = vad
        self.linearout = linearout
        self.window = torch.hamming_window(640, periodic=False).cuda()
        self.freezed = False

    def freeze(self):
        if not self.freezed:
            for param in self.stage1.parameters():
                param.requires_grad = False
            self.freezed = True
            print('freezed stage1')

    def forward(self, feat, mixture, ctl=None):
        if ctl == 'off':
            x = feat
            for i in range(len(self.stage1)):
                x = self.stage1[i](x)
            return x
        else:
            self.freeze()
            x = feat
            for i in range(len(self.stage1)):
                x = self.stage1[i](x)

            spec = torch.stft(
                mixture / 32768,
                640,
                320,
                640,
                self.window,
                center=False,
                return_complex=True)
            spec = torch.view_as_real(spec).permute([0, 2, 1, 3])
            specmag = torch.sqrt(spec[..., 0]**2 + spec[..., 1]**2)
            est = x * specmag
            y = torch.cat([est, feat], dim=-1)
            for i in range(len(self.stage2)):
                y = self.stage2[i](y)
            return y


class Unet(nn.Module):

    def __init__(self,
                 indim,
                 outdim,
                 layers=9,
                 dims=[256] * 4,
                 lorder=20,
                 rorder=0,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 crm=False,
                 vad=False,
                 linearout=False):
        super(Unet, self).__init__()

        self.linear1 = AffineTransform(indim, dims[0])
        self.relu = RectifiedLinear(dims[0], dims[0])

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for i in range(len(dims) - 1):
            layer = nn.Sequential(
                nn.Linear(dims[i], dims[i + 1]), nn.ReLU(),
                nn.Linear(dims[i + 1], dims[i + 1], bias=False),
                Conv2d(
                    dims[i + 1],
                    dims[i + 1],
                    lorder,
                    groups=dims[i + 1],
                    skip_connect=True))
            self.encoder.append(layer)
        for i in range(len(dims) - 1, 0, -1):
            layer = nn.Sequential(
                nn.Linear(dims[i] * 2, dims[i - 1]), nn.ReLU(),
                nn.Linear(dims[i - 1], dims[i - 1], bias=False),
                Conv2d(
                    dims[i - 1],
                    dims[i - 1],
                    lorder,
                    groups=dims[i - 1],
                    skip_connect=True))
            self.decoder.append(layer)
        self.tf = nn.ModuleList()
        for i in range(layers - 2 * (len(dims) - 1)):
            layer = nn.Sequential(
                nn.Linear(dims[-1], dims[-1]), nn.ReLU(),
                nn.Linear(dims[-1], dims[-1], bias=False),
                Conv2d(
                    dims[-1],
                    dims[-1],
                    lorder,
                    groups=dims[-1],
                    skip_connect=True))
            self.tf.append(layer)

        self.linear2 = AffineTransform(dims[0], outdim)
        self.crm = crm
        self.act = nn.Tanh() if self.crm else nn.Sigmoid()
        self.vad = False
        self.layers = layers
        self.linearout = linearout

    def forward(self, x, ctl=None):
        x = self.linear1(x)
        x = self.relu(x)

        encoder_out = []
        for i in range(len(self.encoder)):
            x = self.encoder[i](x)
            encoder_out.append(x)
        for i in range(len(self.tf)):
            x = self.tf[i](x)
        for i in range(len(self.decoder)):
            x = torch.cat([x, encoder_out[-1 - i]], dim=-1)
            x = self.decoder[i](x)

        x = self.linear2(x)
        if self.linearout:
            return x
        return self.act(x)


class BranchNet(nn.Module):

    def __init__(self,
                 indim,
                 outdim,
                 layers=9,
                 hidden_dim=256,
                 lorder=20,
                 rorder=0,
                 dilation=1,
                 layer_norm=False,
                 dropout=0,
                 crm=False,
                 vad=False,
                 linearout=False):
        super(BranchNet, self).__init__()

        self.linear1 = AffineTransform(indim, hidden_dim)
        self.relu = RectifiedLinear(hidden_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.deepfsmn = nn.ModuleList()
        self.FREQ = nn.ModuleList()
        self.TIME = nn.ModuleList()
        self.br1 = nn.ModuleList()
        self.br2 = nn.ModuleList()
        for i in range(layers):
            '''
            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                Conv2d(hidden_dim, hidden_dim, lorder,
                       groups=hidden_dim, skip_connect=True)
            )
            self.deepfsmn.append(layer)
            '''
            layer = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            self.FREQ.append(layer)
            '''
            layer = nn.GRU(hidden_dim, hidden_dim,
                           batch_first=True,
                           bidirectional=False)
            self.TIME.append(layer)

            layer = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim//2, bias=False),
                Conv2d(hidden_dim//2, hidden_dim//2, lorder,
                       groups=hidden_dim//2, skip_connect=True)
            )
            self.br1.append(layer)
            layer = nn.GRU(hidden_dim, hidden_dim//2,
                           batch_first=True,
                           bidirectional=False)
            self.br2.append(layer)
            '''

        self.linear2 = AffineTransform(hidden_dim, outdim)
        self.crm = crm
        self.act = nn.Tanh() if self.crm else nn.Sigmoid()
        self.vad = False
        self.layers = layers
        self.linearout = linearout

    def forward(self, x, ctl=None):
        return self.forward_branch(x)

    def forward_sepconv(self, x):
        x = torch.unsqueeze(x, 1)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            x = F.relu(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3)
        x = torch.reshape(x, [B, H, C * W])
        x = self.linear1(x)
        x = self.relu(x)
        for i in range(self.layers):
            x = self.deepfsmn[i](x) + x
        x = self.linear2(x)
        return self.act(x)

    def forward_branch(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        for i in range(self.layers):
            z = self.FREQ[i](x)
            x = z + x
        x = self.linear2(x)
        if self.linearout:
            return x
        return self.act(x)


class TACNet(nn.Module):
    ''' transform average concatenate for ad hoc dr
    '''

    def __init__(self,
                 indim,
                 outdim,
                 layers=9,
                 hidden_dim=128,
                 lorder=20,
                 rorder=0,
                 crm=False,
                 vad=False,
                 linearout=False):
        super(TACNet, self).__init__()

        self.linear1 = AffineTransform(indim, hidden_dim)
        self.relu = RectifiedLinear(hidden_dim, hidden_dim)

        if rorder == 0:
            repeats = [
                UniDeepFsmn(hidden_dim, hidden_dim, lorder, hidden_dim)
                for i in range(layers)
            ]
        else:
            repeats = [
                DeepFsmn(hidden_dim, hidden_dim, lorder, rorder, hidden_dim)
                for i in range(layers)
            ]
        self.deepfsmn = nn.Sequential(*repeats)

        self.ch_transform = nn.ModuleList([])
        self.ch_average = nn.ModuleList([])
        self.ch_concat = nn.ModuleList([])
        for i in range(layers):
            self.ch_transform.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU()))
            self.ch_average.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.PReLU()))
            self.ch_concat.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim), nn.PReLU()))

        self.linear2 = AffineTransform(hidden_dim, outdim)

        self.crm = crm
        if self.crm:
            self.sig = nn.Tanh()
        else:
            self.sig = Sigmoid(outdim, outdim)

        self.vad = vad
        if self.vad:
            self.linear3 = AffineTransform(hidden_dim, 1)

        self.layers = layers
        self.linearout = linearout
        if self.linearout and self.vad:
            print('Warning: not supported nnet')

    def forward(self, feat, ctl=None):
        B, T, F = feat.shape
        # assume 4ch
        ch = 4
        zlist = []
        for c in range(ch):
            z = self.linear1(feat[..., c * (F // 4):(c + 1) * (F // 4)])
            z = self.relu(z)
            zlist.append(z)
        for i in range(self.layers):
            # forward
            for c in range(ch):
                zlist[c] = self.deepfsmn[i](zlist[c])

            # transform
            olist = []
            for c in range(ch):
                z = self.ch_transform[i](zlist[c])
                olist.append(z)
            # average
            avg = 0
            for c in range(ch):
                avg = avg + olist[c]
            avg = avg / ch
            avg = self.ch_average[i](avg)
            # concate
            for c in range(ch):
                tac = torch.cat([olist[c], avg], dim=-1)
                tac = self.ch_concat[i](tac)
                zlist[c] = zlist[c] + tac

        for c in range(ch):
            zlist[c] = self.sig(self.linear2(zlist[c]))
        mask = torch.cat(zlist, dim=-1)
        return mask

    def to_kaldi_nnet(self):
        pass
