# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
from typing import Any, Dict

import cv2
import mmcv
import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_instance_segmentation.video_knet import \
    KNetTrack
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_instance_segmentation,
    module_name=Pipelines.video_instance_segmentation)
class VideoInstanceSegmentationPipeline(Pipeline):
    r""" Video Instance Segmentation Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline

    >>> detector = pipeline('video-instance-segmentation', 'damo/cv_swinb_video-instance-segmentation')
    >>> detector("http://www.modelscope.cn/api/v1/models/damo/cv_swinb_video-instance-segmentation/repo?Revision=master"
    >>>             "&FilePath=resources/kitti-step_testing_image_02_0000.mp4")
    >>>   {
    >>>    "boxes": [
    >>>        [
    >>>            [
    >>>            0,
    >>>            446.9007568359375,
    >>>            36.374977111816406,
    >>>            907.0919189453125,
    >>>            337.439208984375,
    >>>            0.333
    >>>            ],
    >>>            [
    >>>            1,
    >>>            454.3310241699219,
    >>>            336.08477783203125,
    >>>            921.26904296875,
    >>>            641.7871704101562,
    >>>            0.792
    >>>            ]
    >>>        ],
    >>>        [
    >>>            [
    >>>            0,
    >>>            446.9007568359375,
    >>>            36.374977111816406,
    >>>            907.0919189453125,
    >>>            337.439208984375,
    >>>            0.333
    >>>            ],
    >>>            [
    >>>            1,
    >>>            454.3310241699219,
    >>>            336.08477783203125,
    >>>            921.26904296875,
    >>>            641.7871704101562,
    >>>            0.792
    >>>            ]
    >>>        ]
    >>>    ],
    >>>    "masks": [
    >>>        [
    >>>            [
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            ...,
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False]
    >>>            ],
    >>>            [
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            ...,
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False]
    >>>            ]
    >>>        ],
    >>>        [
    >>>            [
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            ...,
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False]
    >>>            ],
    >>>            [
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            ...,
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False],
    >>>            [False, False, False, ..., False, False, False]
    >>>            ]
    >>>        ]
    >>>    ]
    >>>   }
    >>>
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video panoptic segmentation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        logger.info(f'loading model from {model}')
        model_path = osp.join(model, ModelFile.TORCH_MODEL_FILE)
        config_path = osp.join(model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)
        self.max_video_frames = kwargs.get('max_video_frames', 1000)

        self.model = KNetTrack(model)
        checkpoint = torch.load(
            model_path, map_location=torch.device(self.device))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to(self.device).eval()
        logger.info('load model done')

        self.pad_size_divisor = 32
        self.mean = np.array([123.675, 116.28, 103.53], np.float32)
        self.std = np.array([58.395, 57.12, 57.375], np.float32)
        self.to_rgb = False

    def preprocess(self, input: Input) -> Dict[str, Any]:
        """
         Read video and process into 'imgs', 'img_metas', 'ref_img', 'ref_img_metas'
        """

        if not isinstance(input, str):
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        imgs = []
        img_metas = []

        ref_imgs = []
        ref_img_metas = []

        cap = cv2.VideoCapture(input)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_idx = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx > self.max_video_frames:
                break

            resize_frame = mmcv.imresize(frame, (640, 360))
            norm_frame = mmcv.imnormalize(resize_frame, self.mean, self.std,
                                          self.to_rgb)
            pad_frame = mmcv.impad_to_multiple(
                norm_frame, self.pad_size_divisor, pad_val=0)

            ref_img_meta = {
                'flip': False,
                'flip_direction': None,
                'img_norm_cfg': {
                    'mean': np.array([123.675, 116.28, 103.53],
                                     dtype=np.float32),
                    'std': np.array([58.395, 57.12, 57.375], dtype=np.float32),
                    'to_rgb': True
                },
                'video_id': 0,
                'is_video_data': True
            }
            ref_img_meta['ori_shape'] = frame.shape
            ref_img_meta['img_shape'] = resize_frame.shape
            ref_img_meta['pad_shape'] = pad_frame.shape
            ref_img_meta['frame_id'] = frame_idx

            if frame_idx == 0:
                imgs = [
                    torch.from_numpy(
                        np.array([np.transpose(pad_frame,
                                               [2, 0, 1])])).to(self.device)
                ]
                img_metas = [[ref_img_meta]]

            ref_imgs.append(np.transpose(pad_frame, [2, 0, 1]))
            ref_img_metas.append(ref_img_meta)

            frame_idx += 1

        ref_imgs = np.array([[ref_imgs]])
        ref_img_metas = [[ref_img_metas]]

        result = {
            'video_name': input,
            'imgs': imgs,
            'img_metas': img_metas,
            'ref_img': torch.from_numpy(ref_imgs).to(self.device),
            'ref_img_metas': ref_img_metas,
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
         Segmentation Instance (bounding boxes or masks) in the video passed as inputs.

         Args:
             input (`Video`):
                 The pipeline handles two types of images:

                 - A string containing an HTTP(S) link pointing to a video
                 - A string containing a local path to a video

                 The pipeline accepts a single video as input.


         Return:
             A dictionary of result. If the input is a video, a dictionary
             is returned.

             The dictionary contain the following keys:

             - **boxes** (`List[float]) -- The bounding boxes [index, x1, y1, x2, y2, score] of instance in each frame.
             - **masks** (`List[List[bool]]`, optional) -- The instance mask [[False,...,False],...,[False,...,False]]
         """

        bbox_results = []
        mask_results = []

        with torch.no_grad():
            imgs = input['imgs']
            img_metas = input['img_metas']
            ref_img = input['ref_img']
            ref_img_metas = input['ref_img_metas']

            segm_results = self.model(
                imgs, img_metas, ref_img=ref_img, ref_img_metas=ref_img_metas)

            for ii in range(len(segm_results[0])):
                bbox_results.append(segm_results[0][ii][0])
                mask_results.append(segm_results[0][ii][1])

        output = {
            'boxes': bbox_results,
            'masks': mask_results,
        }
        return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
