# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import os.path as osp
from typing import Any, Dict

import decord
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
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.live_category, module_name=Pipelines.live_category)
class LiveCategoryPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a live-category pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        self.infer_model = models.resnet50(pretrained=False)
        self.infer_model.fc = nn.Linear(2048, 8613)
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.infer_model = self.infer_model.to(self.device).eval()
        self.infer_model.load_state_dict(
            torch.load(model_path, map_location=self.device))
        logger.info('load model done')
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        self.label_mapping = self.cfg.label_mapping
        logger.info('load config done')
        self.transforms = VCompose([
            VRescale(size=256),
            VCenterCrop(size=224),
            VToTensor(),
            VNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            decord.bridge.set_bridge('native')
            vr = VideoReader(input, ctx=cpu(0))
            indices = np.linspace(0, len(vr) - 1, 4).astype(int)
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
        logits = self.infer_model(input['video_data'].to(self.device))
        softmax_out = F.softmax(logits, dim=1).mean(dim=0).cpu()
        scores, ids = softmax_out.topk(3, 0, True, True)
        scores = scores.numpy()
        ids = ids.numpy()
        labels = []
        for i in ids:
            label_info = self.label_mapping[str(i)]
            label_keys = ['cate_level1_name', 'cate_level2_name', 'cate_name']
            label_str = []
            for label_key in label_keys:
                if label_info[label_key] not in label_str:
                    label_str.append(label_info[label_key])
            labels.append(label_str[-1])
        return {OutputKeys.SCORES: list(scores), OutputKeys.LABELS: labels}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


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
        vclip = [
            u.resize((self.size, self.size), Image.BILINEAR) for u in vclip
        ]
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
