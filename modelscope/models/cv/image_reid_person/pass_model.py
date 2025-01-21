# The implementation is adopted from PASS-reID, made publicly available under the Apache-2.0 License at
# https://github.com/CASIA-IVA-Lab/PASS-reID

import os
from enum import Enum

import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .transreid_model import vit_base_patch16_224_TransReID


class Fusions(Enum):
    CAT = 'cat'
    MEAN = 'mean'


@MODELS.register_module(
    Tasks.image_reid_person, module_name=Models.image_reid_person)
class PASS(TorchModel):

    def __init__(self, cfg: Config, model_dir: str, **kwargs):
        super(PASS, self).__init__(model_dir=model_dir)
        size_train = cfg.INPUT.SIZE_TRAIN
        sie_coe = cfg.MODEL.SIE_COE
        stride_size = cfg.MODEL.STRIDE_SIZE
        drop_path = cfg.MODEL.DROP_PATH
        drop_out = cfg.MODEL.DROP_OUT
        att_drop_rate = cfg.MODEL.ATT_DROP_RATE
        gem_pooling = cfg.MODEL.GEM_POOLING
        stem_conv = cfg.MODEL.STEM_CONV
        weight = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.dropout_rate = cfg.MODEL.DROPOUT_RATE
        self.num_classes = cfg.DATASETS.NUM_CLASSES
        self.multi_neck = cfg.MODEL.MULTI_NECK
        self.feat_fusion = cfg.MODEL.FEAT_FUSION

        self.base = vit_base_patch16_224_TransReID(
            img_size=size_train,
            sie_xishu=sie_coe,
            stride_size=stride_size,
            drop_path_rate=drop_path,
            drop_rate=drop_out,
            attn_drop_rate=att_drop_rate,
            gem_pool=gem_pooling,
            stem_conv=stem_conv)
        self.in_planes = self.base.in_planes

        if self.feat_fusion == Fusions.CAT.value:
            self.classifier = nn.Linear(
                self.in_planes * 2, self.num_classes, bias=False)
        elif self.feat_fusion == Fusions.MEAN.value:
            self.classifier = nn.Linear(
                self.in_planes, self.num_classes, bias=False)

        if self.multi_neck:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_1.bias.requires_grad_(False)
            self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_2.bias.requires_grad_(False)
            self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_3.bias.requires_grad_(False)
        else:
            if self.feat_fusion == Fusions.CAT.value:
                self.bottleneck = nn.BatchNorm1d(self.in_planes * 2)
                self.bottleneck.bias.requires_grad_(False)
            elif self.feat_fusion == Fusions.MEAN.value:
                self.bottleneck = nn.BatchNorm1d(self.in_planes)
                self.bottleneck.bias.requires_grad_(False)

        self.dropout = nn.Dropout(self.dropout_rate)

        self.load_param(weight)

    def forward(self, input):

        global_feat, local_feat_1, local_feat_2, local_feat_3 = self.base(
            input)

        # single-neck, almost the same performance
        if not self.multi_neck:
            if self.feat_fusion == Fusions.MEAN.value:
                local_feat = local_feat_1 / 3. + local_feat_2 / 3. + local_feat_3 / 3.
                final_feat_before = (global_feat + local_feat) / 2
            elif self.feat_fusion == Fusions.CAT.value:
                final_feat_before = torch.cat(
                    (global_feat, local_feat_1 / 3. + local_feat_2 / 3.
                     + local_feat_3 / 3.),
                    dim=1)

            final_feat_after = self.bottleneck(final_feat_before)
        # multi-neck
        else:
            feat = self.bottleneck(global_feat)
            local_feat_1_bn = self.bottleneck_1(local_feat_1)
            local_feat_2_bn = self.bottleneck_2(local_feat_2)
            local_feat_3_bn = self.bottleneck_3(local_feat_3)

            if self.feat_fusion == Fusions.MEAN.value:
                final_feat_before = ((global_feat + local_feat_1 / 3
                                      + local_feat_2 / 3 + local_feat_3 / 3)
                                     / 2.)
                final_feat_after = (feat + local_feat_1_bn / 3
                                    + local_feat_2_bn / 3
                                    + local_feat_3_bn / 3) / 2.
            elif self.feat_fusion == Fusions.CAT.value:
                final_feat_before = torch.cat(
                    (global_feat, local_feat_1 / 3. + local_feat_2 / 3.
                     + local_feat_3 / 3.),
                    dim=1)
                final_feat_after = torch.cat(
                    (feat, local_feat_1_bn / 3 + local_feat_2_bn / 3
                     + local_feat_3_bn / 3),
                    dim=1)

        if self.neck_feat == 'after':
            return final_feat_after
        else:
            return final_feat_before

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        for i in param_dict:
            try:
                self.state_dict()[i.replace('module.',
                                            '')].copy_(param_dict[i])
            except Exception:
                continue
