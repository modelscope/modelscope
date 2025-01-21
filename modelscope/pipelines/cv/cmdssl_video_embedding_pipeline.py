# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os.path as osp
from typing import Any, Dict

import decord
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models.cv.cmdssl_video_embedding import resnet26_2p1d
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_embedding, module_name=Pipelines.cmdssl_video_embedding)
class CMDSSLVideoEmbeddingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a CMDSSL Video Embedding pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        self.model = resnet26_2p1d(num_classes=None, last_pool=True)

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.model = self.model.to(self._device).eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(model_path))
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        decord.bridge.set_bridge('native')

        transforms = VCompose([
            VRescale(size=self.cfg.DATA.scale_size),
            VCenterCrop(size=self.cfg.DATA.crop_size),
            VToTensor(),
            VNormalize(mean=self.cfg.DATA.mean, std=self.cfg.DATA.std)
        ])

        clip_len = (self.cfg.DATA.video_frames
                    - 1) * self.cfg.DATA.video_stride + 1
        vr = decord.VideoReader(input, ctx=decord.cpu(0))
        if len(vr) <= clip_len:
            init_frames = np.zeros(self.cfg.DATA.multi_crop, dtype=int)
        else:
            init_frames = np.linspace(0,
                                      len(vr) - clip_len,
                                      self.cfg.DATA.multi_crop + 1)
            init_frames = ((init_frames[1:] + init_frames[:-1])
                           / 2.).astype(int)

        indices = np.arange(0, clip_len, self.cfg.DATA.video_stride)
        indices = (init_frames[:, None] + indices[None, :]).reshape(-1)
        indices[indices >= len(vr)] = 0

        frames = torch.from_numpy(vr.get_batch(indices).asnumpy()).chunk(
            self.cfg.DATA.multi_crop, dim=0)
        frames = [
            transforms([Image.fromarray(f) for f in u.numpy()]) for u in frames
        ]
        frames = torch.stack(frames, dim=0)
        result = {'video_data': frames}
        return result

    @torch.no_grad()
    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        frames = input['video_data'].to(self._device)
        feature = self.model(frames)
        feature = feature.mean(0)
        return {OutputKeys.VIDEO_EMBEDDING: feature.data.cpu().numpy()}

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
        vclip = torch.stack([TF.to_tensor(u) for u in vclip], dim=1)
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
            self.mean = vclip.new_tensor(self.mean).view(-1, 1, 1, 1)
        if not isinstance(self.std, torch.Tensor):
            self.std = vclip.new_tensor(self.std).view(-1, 1, 1, 1)
        vclip.sub_(self.mean).div_(self.std)
        return vclip
