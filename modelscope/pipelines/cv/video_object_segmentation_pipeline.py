# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_object_segmentation.inference_core import \
    InferenceCore
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()

im_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def unpad(img, pad):
    if pad[2] + pad[3] > 0:
        img = img[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        img = img[:, :, :, pad[0]:-pad[1]]
    return img


def all_to_onehot(masks, labels):
    if len(masks.shape) == 3:
        Ms = np.zeros(
            (len(labels), masks.shape[0], masks.shape[1], masks.shape[2]),
            dtype=np.uint8)
    else:
        Ms = np.zeros((len(labels), masks.shape[0], masks.shape[1]),
                      dtype=np.uint8)

    for k, l in enumerate(labels):
        Ms[k] = (masks == l).astype(np.uint8)

    return Ms


@PIPELINES.register_module(
    Tasks.video_object_segmentation,
    module_name=Pipelines.video_object_segmentation)
class VideoObjectSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create video_object_segmentation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """

        super().__init__(model=model, **kwargs)
        logger.info('load model done')
        self.im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
            transforms.Resize(480, interpolation=Image.BICUBIC),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(480, interpolation=Image.NEAREST),
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:

        self.images = input['images']
        self.mask = input['mask']

        frames = len(self.images)
        shape = np.shape(self.mask)

        info = {}
        info['name'] = 'maas_test_video'
        info['frames'] = frames
        info['size'] = shape  # Real sizes
        info['gt_obj'] = {}  # Frames with labelled objects

        images = []
        masks = []
        for i in range(frames):
            img = self.images[i]
            images.append(self.im_transform(img))

            palette = self.mask.getpalette()
            masks.append(np.array(self.mask, dtype=np.uint8))
            this_labels = np.unique(masks[-1])
            this_labels = this_labels[this_labels != 0]
            info['gt_obj'][i] = this_labels

        images = torch.stack(images, 0)
        masks = np.stack(masks, 0)

        labels = np.unique(masks).astype(np.uint8)
        labels = labels[labels != 0]

        masks = torch.from_numpy(all_to_onehot(masks, labels)).float()
        # Resize to 480p
        masks = self.mask_transform(masks)
        masks = masks.unsqueeze(2)

        info['labels'] = labels

        result = {
            'rgb': images,
            'gt': masks,
            'info': info,
            'palette': np.array(palette),
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        rgb = input['rgb'].unsqueeze(0)
        msk = input['gt']
        info = input['info']
        k = len(info['labels'])
        size = info['size']

        is_cuda = rgb.is_cuda

        processor = InferenceCore(
            self.model.model, is_cuda, rgb, k, top_k=20, mem_every=4)
        processor.interact(msk[:, 0], 0, rgb.shape[1])

        # Do unpad -> upsample to original size
        out_masks = torch.zeros((processor.t, 1, *size),
                                dtype=torch.uint8,
                                device='cuda' if is_cuda else 'cpu')
        for ti in range(processor.t):
            prob = unpad(processor.prob[:, ti], processor.pad)
            prob = F.interpolate(
                prob, tuple(size), mode='bilinear', align_corners=False)
            out_masks[ti] = torch.argmax(prob, dim=0)

        if is_cuda:
            out_masks = (out_masks.detach().cpu().numpy()[:,
                                                          0]).astype(np.uint8)
        else:
            out_masks = (out_masks.detach().numpy()[:, 0]).astype(np.uint8)

        return {OutputKeys.MASKS: out_masks}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
