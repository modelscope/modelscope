# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os.path as osp
from typing import Any, Dict

import json
import torch
import torch.nn as nn
import torch.nn.functional as F

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from .backbone import build_backbone
from .head import FPNSegmentor, LinearClassifier


@MODELS.register_module(
    Tasks.image_segmentation, module_name=Models.vision_middleware)
class VisionMiddlewareModel(TorchModel):
    """
        The implementation of 'ViM: Vision Middleware for Unified Downstream Transferring'.
        This model is dynamically initialized with the following parts:
            - backbone: the upstream pre-trained backbone model (CLIP in this code)
            - ViM: the zoo of middlestream trained ViM modules
            - ViM-aggregation: the specific aggregation weights for downstream tasks
    """

    def __init__(self, model_dir: str, *args, **kwargs):
        """
            Initialize a ViM-based Model
            Args:
                model_dir: model id or path,
                where model_dir/pytorch_model.pt contains:
                    'meta_info': basic information of ViM, e.g. task_list
                    'backbone_weights': parameters of backbone [upstream]
                    'ViM_weights': parameters of ViM [midstream]
                    'ViM_agg_weights': parameters of ViM-aggregation [downstream]
        """
        super(VisionMiddlewareModel, self).__init__()

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        model_dict = torch.load(model_path, map_location='cpu')

        meta_info = model_dict['meta_info']
        self.task_list = meta_info['task_list']

        # build up backbone
        backbone_weights = model_dict['backbone_weights']
        self.backbone = build_backbone(
            arch=meta_info['backbone_arch'], pretrained=backbone_weights)
        self.backbone.eval()

        # build up ViM
        vim_weights = model_dict['ViM_weights']
        num_layers = len(vim_weights)
        for layer_i in range(num_layers):
            self.backbone.transformer.resblocks[layer_i].vim_att.register_ViM(
                vim_weights[layer_i]['vim_att_weights'])
            self.backbone.transformer.resblocks[layer_i].vim_mlp.register_ViM(
                vim_weights[layer_i]['vim_mlp_weights'])

        # build up each task-related ViM aggregation
        agg_weights = model_dict['ViM_agg_weights']
        agg_algo = meta_info['ViM_agg_algo']
        for task_name in meta_info['task_list']:
            for layer_i in range(num_layers):
                self.backbone.transformer.resblocks[
                    layer_i].vim_att.register_task(
                        task_name,
                        agg_weights[task_name][layer_i]['vim_att_agg'],
                        agg_algo)
                self.backbone.transformer.resblocks[
                    layer_i].vim_mlp.register_task(
                        task_name,
                        agg_weights[task_name][layer_i]['vim_mlp_agg'],
                        agg_algo)

        # build up each task-related head
        self.heads = nn.ModuleDict()
        self.label_maps = {}
        for task_name in meta_info['task_list']:
            head_weights = model_dict['head_weights']
            if task_name.startswith('cls'):
                self.heads[task_name] = LinearClassifier(
                    in_channels=self.backbone.output_dim,
                    num_classes=head_weights[task_name]
                    ['classifier.bias'].shape[0])
            elif task_name.startswith('seg'):
                self.heads[task_name] = FPNSegmentor()
            else:
                raise NotImplementedError(
                    'Task type [{}] is not supported'.format(task_name))

            self.heads[task_name].load_state_dict(head_weights[task_name])
            self.heads[task_name].eval()

            if task_name in meta_info['label_map'].keys():
                self.label_maps[task_name] = meta_info['label_map'][task_name]

    def __call__(self, inputs, task_name) -> Dict[str, Any]:
        return self.postprocess(
            self.forward(inputs, task_name), inputs, task_name)

    def forward(self, inputs, task_name):
        """
            Dynamic Forward Function of ViM
            Args:
                x: the input images (B, 3, H, W)
                task_name: specified task for forwarding
        """
        if task_name not in self.task_list:
            raise NotImplementedError(
                f'task_name should in {self.task_list}, but got {task_name}')

        features = self.backbone(inputs, task_name=task_name)
        outputs = self.heads[task_name](features)

        return outputs

    def postprocess(self, outputs, inputs, task_name):
        """
            Post-process of ViM, based on task_name
            Args:
                inputs: batched input image (B, 3, H, W)
                outputs: batched output (format based on task_name)
                task_name: str, task name
        """

        _, in_channels, img_height, img_width = inputs.size()

        if 'seg' in task_name:
            # outputs in shape of [1, C, H, W]
            seg = F.softmax(outputs, dim=1)
            seg = F.interpolate(seg, (img_height, img_width), None, 'bilinear',
                                True)
            seg = seg[0].detach().cpu()
            pred = torch.argmax(seg, dim=0)

            labels = sorted(list(set(pred.reshape(-1).numpy())))

            masks, scores = [], []
            for label in labels:
                mask = (pred == label)
                masks.append(mask.long().numpy())
                scores.append(((mask.float() * seg[label]).sum()
                               / mask.float().sum()).item())

            label_names = [
                self.label_maps[task_name][label] for label in labels
            ]

            return {
                OutputKeys.MASKS: masks,
                OutputKeys.LABELS: label_names,
                OutputKeys.SCORES: scores
            }
        else:
            raise NotImplementedError(
                'Only segmentation task is currently supported in pipeline')

    def get_tasks(self):
        """
            Get the supported tasks of current ViM model
        """
        return self.task_list
