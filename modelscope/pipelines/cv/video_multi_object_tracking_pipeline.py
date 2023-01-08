# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_multi_object_tracking.tracker.multitracker import \
    JDETracker
from modelscope.models.cv.video_multi_object_tracking.utils.utils import (
    LoadVideo, cfg_opt)
from modelscope.models.cv.video_single_object_tracking.utils.utils import \
    timestamp_format
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_multi_object_tracking,
    module_name=Pipelines.video_multi_object_tracking)
class VideoMultiObjectTrackingPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a multi object tracking pipeline
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        ckpt_path = osp.join(model, ModelFile.TORCH_MODEL_BIN_FILE)
        logger.info(f'loading model from {ckpt_path}')
        opt = cfg_opt()
        self.opt = opt
        self.tracker = JDETracker(opt, ckpt_path, self.device)
        logger.info('init tracker done')

    def preprocess(self, input) -> Input:
        self.video_path = input[0]
        return input

    def forward(self, input: Input) -> Dict[str, Any]:
        dataloader = LoadVideo(input, self.opt.img_size)
        self.tracker.set_buffer_len(dataloader.frame_rate)

        results = []
        output_timestamps = []
        frame_id = 0
        for i, (path, img, img0) in enumerate(dataloader):
            output_timestamps.append(
                timestamp_format(seconds=frame_id / dataloader.frame_rate))
            blob = torch.from_numpy(img).unsqueeze(0)
            online_targets = self.tracker.update(blob, img0)
            online_tlwhs = []
            online_ids = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > self.opt.min_box_area and not vertical:
                    online_tlwhs.append([
                        tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    ])
                    online_ids.append(tid)
                results.append([
                    frame_id + 1, tid, tlwh[0], tlwh[1], tlwh[0] + tlwh[2],
                    tlwh[1] + tlwh[3]
                ])
            frame_id += 1

        return {
            OutputKeys.BOXES: results,
            OutputKeys.TIMESTAMPS: output_timestamps
        }

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
