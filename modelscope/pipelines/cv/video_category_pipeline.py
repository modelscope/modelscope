# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os.path as osp
from typing import Any, Dict

import decord
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from decord import VideoReader, cpu
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_category, module_name=Pipelines.video_category)
class VideoCategoryPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video-category pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading configuration from {config_path}')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            self.frame_num = config['frame_num']
            self.level_1_num = config['level_1_num']
            self.level_2_num = config['level_2_num']
            self.resize = config['resize']
            self.crop = config['crop']
            self.mean = config['mean']
            self.std = config['std']
            self.cateproj_v3 = config['cateproj_v3']
            self.class_name = config['class_name']
            self.subclass_name = config['subclass_name']
        logger.info('load configuration done')

        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        self.infer_model = ModelWrapper(self.level_1_num, self.level_2_num,
                                        self.frame_num)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = self.infer_model.to(self.device).eval()
        self.infer_model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        logger.info('load model done')
        self.transforms = VCompose([
            VRescale(size=self.resize),
            VCenterCrop(size=self.crop),
            VToTensor(),
            VNormalize(mean=self.mean, std=self.std)
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            decord.bridge.set_bridge('native')
            vr = VideoReader(input, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, 16).astype(int)
            frames = vr.get_batch(indices).asnumpy()
            video_input_data = self.transforms(
                [Image.fromarray(f) for f in frames])
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        result = {'video_data': video_input_data}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        pred1, pred2 = self.infer_model(input['video_data'].to(self.device))

        pred1 = F.softmax(pred1, dim=1)
        pred2 = F.softmax(pred2, dim=1)

        vals_2, preds_2 = pred2.cpu().topk(10, 1, True, True)
        vals_2 = vals_2.detach().numpy()
        preds_2 = preds_2.detach().numpy()

        if vals_2[0][0] >= 0.3:
            c2 = int(preds_2[0][0])
            c1 = self.cateproj_v3[c2]

            tag1 = self.class_name[c1]
            tag2 = self.subclass_name[c2]

            prob = float(vals_2[0][0])
        else:
            vals_1, preds_1 = pred1.cpu().topk(10, 1, True, True)
            vals_1 = vals_1.detach().numpy()
            preds_1 = preds_1.detach().numpy()

            c1 = int(preds_1[0][0])

            tag1 = self.class_name[c1]
            tag2 = '其他'

            prob = float(vals_1[0][0])

        return {
            OutputKeys.SCORES: [prob],
            OutputKeys.LABELS: [tag1 + '>>' + tag2]
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


class TimeFirstBatchNorm1d(nn.Module):

    def __init__(self, dim, groups=None):
        super().__init__()
        self.groups = groups
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, tensor):
        _, length, dim = tensor.size()
        if self.groups:
            dim = dim // self.groups
        tensor = tensor.view(-1, dim)
        tensor = self.bn(tensor)
        if self.groups:
            return tensor.view(-1, length, self.groups, dim)
        else:
            return tensor.view(-1, length, dim)


class NeXtVLAD(nn.Module):
    """NeXtVLAD layer implementation
    Adapted from https://github.com/linrongc/youtube-8m/blob/master/nextvlad.py
    """

    def __init__(self,
                 num_clusters=64,
                 dim=128,
                 alpha=100.0,
                 groups=8,
                 expansion=2,
                 normalize_input=True,
                 p_drop=0.25,
                 add_batchnorm=False):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NeXtVLAD, self).__init__()
        assert dim % groups == 0, '`dim` must be divisible by `groups`'
        assert expansion > 1
        self.p_drop = p_drop
        self.cluster_dropout = nn.Dropout2d(p_drop)
        self.num_clusters = num_clusters
        self.dim = dim
        self.expansion = expansion
        self.grouped_dim = dim * expansion // groups
        self.groups = groups
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.add_batchnorm = add_batchnorm
        self.expansion_mapper = nn.Linear(dim, dim * expansion)
        if add_batchnorm:
            self.soft_assignment_mapper = nn.Sequential(
                nn.Linear(dim * expansion, num_clusters * groups, bias=False),
                TimeFirstBatchNorm1d(num_clusters, groups=groups))
        else:
            self.soft_assignment_mapper = nn.Linear(
                dim * expansion, num_clusters * groups, bias=True)
        self.attention_mapper = nn.Linear(dim * expansion, groups)
        self.centroids = nn.Parameter(
            torch.rand(num_clusters, self.grouped_dim))
        self.final_bn = nn.BatchNorm1d(num_clusters * self.grouped_dim)
        self._init_params()

    def _init_params(self):
        for component in (self.soft_assignment_mapper, self.attention_mapper,
                          self.expansion_mapper):
            for module in component.modules():
                self.general_weight_initialization(module)
        if self.add_batchnorm:
            self.soft_assignment_mapper[0].weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat(
                    (self.groups, self.groups)))
            nn.init.constant_(self.soft_assignment_mapper[1].bn.weight, 1)
            nn.init.constant_(self.soft_assignment_mapper[1].bn.bias, 0)
        else:
            self.soft_assignment_mapper.weight = nn.Parameter(
                (2.0 * self.alpha * self.centroids).repeat(
                    (self.groups, self.groups)))
            self.soft_assignment_mapper.bias = nn.Parameter(
                (-self.alpha * self.centroids.norm(dim=1)).repeat(
                    (self.groups, )))

    def general_weight_initialization(self, module):
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if module.weight is not None:
                nn.init.uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, masks=None):
        """NeXtVlad Adaptive Pooling
        Arguments:
            x {torch.Tensor} -- shape: (n_batch, len, dim)
        Returns:
            torch.Tensor -- shape (n_batch, n_cluster * dim / groups)
        """
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=2)  # across descriptor dim

        # expansion
        # shape: (n_batch, len, dim * expansion)
        x = self.expansion_mapper(x)

        # soft-assignment
        # shape: (n_batch, len, n_cluster, groups)
        soft_assign = self.soft_assignment_mapper(x).view(
            x.size(0), x.size(1), self.num_clusters, self.groups)
        soft_assign = F.softmax(soft_assign, dim=2)

        # attention
        # shape: (n_batch, len, groups)
        attention = torch.sigmoid(self.attention_mapper(x))
        if masks is not None:
            # shape: (n_batch, len, groups)
            attention = attention * masks[:, :, None]

        # (n_batch, len, n_cluster, groups, dim / groups)
        activation = (
            attention[:, :, None, :, None] * soft_assign[:, :, :, :, None])

        # calculate residuals to each clusters
        # (n_batch, n_cluster, dim / groups)
        second_term = (
            activation.sum(dim=3).sum(dim=1) * self.centroids[None, :, :])
        # (n_batch, n_cluster, dim / groups)
        first_term = (
            # (n_batch, len, n_cluster, groups, dim / groups)
            activation
            * x.view(x.size(0), x.size(1), 1, self.groups,
                     self.grouped_dim)).sum(dim=3).sum(dim=1)

        # vlad shape (n_batch, n_cluster, dim / groups)
        vlad = first_term - second_term
        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        # flatten shape (n_batch, n_cluster * dim / groups)
        vlad = vlad.view(x.size(0), -1)  # flatten
        # vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize
        vlad = self.final_bn(vlad)
        if self.p_drop:
            vlad = self.cluster_dropout(
                vlad.view(x.size(0), self.num_clusters, self.grouped_dim,
                          1)).view(x.size(0), -1)
        return vlad


class ModelWrapper(nn.Module):

    def __init__(self, class_num, subclass_num, frame_num):
        super(ModelWrapper, self).__init__()
        cnn = models.resnet50(pretrained=False)
        cnn.fc = nn.Sequential()
        self.model = cnn
        # Use NextVlad
        # output size: (n_batch, n_cluster * dim / groups)
        nv_group = 2
        expand = int(2 * frame_num / nv_group)
        self.nextvlad = NeXtVLAD(
            num_clusters=frame_num, dim=2048, groups=nv_group)
        self.fc = nn.Linear(2048 * expand, 2048)
        self.head1_p1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.head1_p2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, class_num),
        )
        self.head2_p1 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        )
        self.head2_p2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, subclass_num),
        )
        self.fn = frame_num

    def forward(self, x):
        x = x.view(-1, 3, 224, 224)
        x = self.model(x)

        x = x.view(-1, self.fn, 2048)
        x = self.nextvlad(x)

        x = self.fc(x)

        x1 = self.head1_p1(x)
        c1 = self.head1_p2(x1)

        x2 = self.head2_p1(x)
        c2 = self.head2_p2(torch.cat((x1, x2), dim=1))

        return c1, c2


class VCompose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, item):
        for t in self.transforms:
            item = t(item)
        return item


class VRescale(object):

    def __init__(self, size=128):
        self.size = size

    def __call__(self, vclip):
        w, h = vclip[0].size
        scale = self.size / min(w, h)
        out_w, out_h = int(round(w * scale)), int(round(h * scale))
        vclip = [u.resize((out_w, out_h), Image.BILINEAR) for u in vclip]
        return vclip


class VCenterCrop(object):

    def __init__(self, size=112):
        self.size = size

    def __call__(self, vclip):
        w, h = vclip[0].size
        assert min(w, h) >= self.size
        x1 = (w - self.size) // 2
        y1 = (h - self.size) // 2
        vclip = [
            u.crop((x1, y1, x1 + self.size, y1 + self.size)) for u in vclip
        ]
        return vclip


class VToTensor(object):

    def __call__(self, vclip):
        vclip = torch.stack([TF.to_tensor(u) for u in vclip], dim=0)
        return vclip


class VNormalize(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, vclip):
        assert vclip.min() > -0.1 and vclip.max() < 1.1, \
            'vclip values should be in [0, 1]'
        vclip = vclip.clone()
        if not isinstance(self.mean, torch.Tensor):
            self.mean = vclip.new_tensor(self.mean).view(1, -1, 1, 1)
        if not isinstance(self.std, torch.Tensor):
            self.std = vclip.new_tensor(self.std).view(1, -1, 1, 1)
        vclip.sub_(self.mean).div_(self.std)
        return vclip
