# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import torch
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_summarization import (PGLVideoSummarization,
                                                      summary_format)
from modelscope.models.cv.video_summarization.base_model import bvlc_googlenet
from modelscope.models.cv.video_summarization.summarizer import (
    generate_summary, get_change_points)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_summarization, module_name=Pipelines.video_summarization)
class VideoSummarizationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video summarization pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        logger.info(f'loading model from {model}')
        googlenet_model_path = osp.join(model, 'bvlc_googlenet.pt')
        config_path = osp.join(model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)

        self.googlenet_model = bvlc_googlenet()
        self.googlenet_model.model.load_state_dict(
            torch.load(
                googlenet_model_path, map_location=torch.device(self.device)))
        self.googlenet_model = self.googlenet_model.to(self.device).eval()

        self.pgl_model = PGLVideoSummarization(model)
        self.pgl_model = self.pgl_model.to(self.device).eval()

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if not isinstance(input, str):
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        frames = []
        picks = []
        cap = cv2.VideoCapture(input)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_idx = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 15 == 0:
                frames.append(frame)
                picks.append(frame_idx)
            frame_idx += 1
        n_frame = frame_idx

        result = {
            'video_name': input,
            'video_frames': np.array(frames),
            'n_frame': n_frame,
            'picks': np.array(picks)
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:

        frame_features = []
        for frame in tqdm(input['video_frames']):
            feat = self.googlenet_model(frame)
            frame_features.append(feat)

        change_points, n_frame_per_seg = get_change_points(
            frame_features, input['n_frame'])

        summary = self.inference(frame_features, input['n_frame'],
                                 input['picks'], change_points)

        output = summary_format(summary, self.fps)

        return {OutputKeys.OUTPUT: output}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def inference(self, frame_features, n_frames, picks, change_points):
        frame_features = torch.from_numpy(np.array(frame_features, np.float32))
        picks = np.array(picks, np.int32)

        with torch.no_grad():
            results = self.pgl_model(dict(frame_features=frame_features))
            scores = results['scores']
            if not scores.device.type == 'cpu':
                scores = scores.cpu()
            scores = scores.squeeze(0).numpy().tolist()
            summary = generate_summary([change_points], [scores], [n_frames],
                                       [picks])[0]

        return summary.tolist()
