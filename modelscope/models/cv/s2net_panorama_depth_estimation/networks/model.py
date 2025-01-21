# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
import torch
import torch.nn as nn

from .decoder import SphDecoder
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
# import backbone
from .swin_transformer import SwinTransformer
from .util_helper import compute_hp_neighmap


class SwinSphDecoderNet(nn.Module):

    def __init__(self, cfg, pretrained=False):
        super(SwinSphDecoderNet, self).__init__()
        # compute healpixel
        self.neighbor_maps_all_levels = compute_hp_neighmap(
            [128, 64, 32, 16, 8, 4])
        self.vit_encoder = SwinTransformer(
            pretrain_img_size=cfg.BACKBONE.PRETRAIN_RES,
            embed_dim=cfg.BACKBONE.EMBED_DIM,
            depths=cfg.BACKBONE.DEPTHS,
            num_heads=cfg.BACKBONE.NUM_HEADS,
            window_size=cfg.BACKBONE.WINDOW_SIZE,
            drop_path_rate=cfg.BACKBONE.DROP_PATH_RATE,
            frozen_stages=cfg.BACKBONE.FROZEN_STAGES)

        ch = cfg.BACKBONE.EMBED_DIM
        self.num_ch_enc = [ch, 2 * ch, 4 * ch, 8 * ch]
        self.decoder = SphDecoder(
            cfg=cfg,
            neighbor_maps=self.neighbor_maps_all_levels,
            num_ch_enc=self.num_ch_enc,
            img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))

        cfg.defrost()
        if pretrained:
            # no offline pretrained model
            if cfg.TRAIN.PRETRAINED_MODEL == '':
                cfg.TRAIN.PRETRAINED_MODEL = 'modelzoo://swin_transformer/swin_' + cfg.BACKBONE.VERSION \
                    + '_patch4_window' + str(cfg.BACKBONE.WINDOW_SIZE) + '_' + str(cfg.BACKBONE.PRETRAIN_RES)
                if cfg.BACKBONE.PRETRAIN_IMAGENET == '1k':
                    cfg.TRAIN.PRETRAINED_MODEL += '.pth'
                elif cfg.BACKBONE.PRETRAIN_IMAGENET == '22k':
                    cfg.TRAIN.PRETRAINED_MODEL += '_22k.pth'
        else:
            cfg.TRAIN.PRETRAINED_MODEL = ''
        cfg.freeze()
        if cfg.TRAIN.PRETRAINED_MODEL != '':
            self.init_weights(cfg.TRAIN.PRETRAINED_MODEL)

    def init_weights(self, pretrained_model):
        self.vit_encoder.init_weights(pretrained_model)

    def forward(self, x):
        out = self.vit_encoder(x)
        out = self.decoder(out[0], out[1], out[2], out[3])
        return out


class ResnetSphDecoderNet(nn.Module):

    def __init__(self, cfg, pretrained=False):
        super(ResNetSphDecoder, self).__init__()
        self.neighbor_maps_all_levels = compute_hp_neighmap(
            [128, 64, 32, 16, 8, 4])
        encoder = {
            18: resnet18,
            34: resnet34,
            50: resnet50,
            101: resnet101,
            152: resnet152
        }
        self.encoder = encoder[cfg.BACKBONE.RESNET_LAYER_NUM](pretrained)

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        if cfg.BACKBONE.RESNET_LAYER_NUM > 34:
            self.num_ch_enc[1:] *= 4

        if cfg.BACKBONE.RESNET_LAYER_NUM < 18:
            self.num_ch_enc = np.array([16, 24, 32, 96, 320])
        self.num_layers = cfg.BACKBONE.RESNET_LAYER_NUM

        self.decoder = SphDecoder(
            cfg=cfg,
            neighbor_maps=self.neighbor_maps_all_levels,
            num_ch_enc=self.num_ch_enc[1:],
            img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))

    def forward(self, x):
        if self.num_layers < 18:
            enc_feat0, enc_feat1, enc_feat2, enc_feat3, enc_feat4 \
                = self.encoder(x)
        else:
            x = self.encoder.conv1(x)
            x = self.encoder.relu(self.encoder.bn1(x))
            x = self.encoder.maxpool(x)
            enc_feat1 = self.encoder.layer1(x)
            enc_feat2 = self.encoder.layer2(enc_feat1)
            enc_feat3 = self.encoder.layer3(enc_feat2)
            enc_feat4 = self.encoder.layer4(enc_feat3)
        out = self.decoder(enc_feat1, enc_feat2, enc_feat3, enc_feat4)
        return out


class EfficientNetEncoder(nn.Module):

    def __init__(self, backend):
        super(EfficientNetEncoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'blocks':
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class EffnetSphDecoderNet(nn.Module):

    def __init__(self, cfg, pretrained=False):
        super(EffNetSphDecoder, self).__init__()
        self.neighbor_maps_all_levels = compute_hp_neighmap(
            [128, 64, 32, 16, 8, 4])
        basemodel_name = 'tf_efficientnet_b5_ap'
        basemodel = torch.hub.load(
            'rwightman/gen-efficientnet-pytorch',
            basemodel_name,
            pretrained=pretrained)
        # Remove last layer
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()
        self.encoder = EfficientNetEncoder(basemodel)

        self.num_ch_enc = np.array([24, 40, 64, 176, 2048])
        self.decoder = SphDecoder(
            cfg=cfg,
            neighbor_maps=self.neighbor_maps_all_levels,
            num_ch_enc=self.num_ch_enc[1:],
            img_size=(cfg.DATA.IMG_HEIGHT, cfg.DATA.IMG_WIDTH))
        self.conv4 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=1)
        self.relu = nn.ReLU(False)

    def forward(self, x):
        enc_feat = self.encoder(x)
        enc_feat1, enc_feat2, enc_feat3, enc_feat4 = \
            enc_feat[5], enc_feat[6], enc_feat[8], enc_feat[11]
        enc_feat4 = self.relu(self.conv4(enc_feat4))
        out = self.decoder(enc_feat1, enc_feat2, enc_feat3, enc_feat4)
        return out
