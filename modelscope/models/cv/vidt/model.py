# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import SwinTransformer
from .deformable_transformer import DeformableTransformer
from .fpn_fusion import FPNFusionModule
from .head import Detector


@MODELS.register_module(Tasks.image_object_detection, module_name=Models.vidt)
class VidtModel(TorchModel):
    """
        The implementation of 'ViDT for joint-learning of object detection and instance segmentation'.
        This model is dynamically initialized with the following parts:
            - 'backbone': pre-trained backbone model with parameters.
            - 'head': detection and segentation head with fine-tuning.
    """

    def __init__(self, model_dir: str, **kwargs):
        """ Initialize a Vidt Model.
        Args:
          model_dir: model id or path, where model_dir/pytorch_model.pt contains:
                    - 'backbone_weights': parameters of backbone.
                    - 'head_weights': parameters of head.
        """
        super(VidtModel, self).__init__()

        model_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        model_dict = torch.load(model_path, map_location='cpu')

        # build backbone
        backbone = SwinTransformer(
            pretrain_img_size=[224, 224],
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            drop_path_rate=0.2)
        backbone.finetune_det(
            method='vidt', det_token_num=300, pos_dim=256, cross_indices=[3])
        self.backbone = backbone
        self.backbone.load_state_dict(
            model_dict['backbone_weights'], strict=True)

        # build head
        epff = FPNFusionModule(backbone.num_channels, fuse_dim=256)
        deform_transformers = DeformableTransformer(
            d_model=256,
            nhead=8,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation='relu',
            return_intermediate_dec=True,
            num_feature_levels=4,
            dec_n_points=4,
            token_label=False)
        head = Detector(
            backbone,
            deform_transformers,
            num_classes=2,
            num_queries=300,
            # two essential techniques used in ViDT
            aux_loss=True,
            with_box_refine=True,
            # an epff module for ViDT+
            epff=epff,
            # an UQR module for ViDT+
            with_vector=False,
            processor_dct=None,
            # two additional losses for VIDT+
            iou_aware=True,
            token_label=False,
            vector_hidden_dim=256,
            # distil
            distil=False)
        self.head = head
        self.head.load_state_dict(model_dict['head_weights'], strict=True)

    def forward(self, x, mask):
        """ Dynamic forward function of VidtModel.
        Args:
            x: input images (B, 3, H, W)
            mask: input padding masks (B, H, W)
        """
        features_0, features_1, features_2, features_3, det_tgt, det_pos = self.backbone(
            x, mask)
        out_pred_logits, out_pred_boxes = self.head(features_0, features_1,
                                                    features_2, features_3,
                                                    det_tgt, det_pos, mask)
        return out_pred_logits, out_pred_boxes
