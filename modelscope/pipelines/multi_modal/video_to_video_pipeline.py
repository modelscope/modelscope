# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

import cv2
import torch
from einops import rearrange

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Model, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.image import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_to_video, module_name=Pipelines.video_to_video_pipeline)
class VideoToVideoPipeline(Pipeline):
    r""" Video To Video Pipeline, generating super-resolution videos based on input
    video and text

    Examples:
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.outputs import OutputKeys

    >>> # YOUR_VIDEO_PATH:   your video url or local position in low resolution
    >>> # INPUT_TEXT:        when we do video super-resolution, we will add the text content
    >>> #                    into results
    >>> # output_video_path: path-to-the-generated-video

    >>> p = pipeline('video-to-video', 'damo/Video-to-Video')
    >>> input = {"video_path":YOUR_VIDEO_PATH, "text": INPUT_TEXT}
    >>> output_video_path = p(input,output_video='./output.mp4')[OutputKeys.OUTPUT_VIDEO]

    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a kws pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self._model_prepare = True

    def preprocess(self, input: Input, **preprocess_params) -> Dict[str, Any]:
        vid_path = input['video_path']
        if 'text' in input.keys():
            text = input['text']
        else:
            text = ''

        caption = text + self.model.positive_prompt
        y = self.model.clip_encoder(caption).detach()

        max_frames = self.model.cfg.max_frames

        capture = cv2.VideoCapture(vid_path)
        _fps = capture.get(cv2.CAP_PROP_FPS)
        sample_fps = _fps
        _total_frame_num = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        stride = round(_fps / sample_fps)
        start_frame = 0

        pointer = 0
        frame_list = []
        while len(frame_list) < max_frames:
            ret, frame = capture.read()
            pointer += 1
            if (not ret) or (frame is None):
                break
            if pointer < start_frame:
                continue
            if pointer >= _total_frame_num + 1:
                break
            if (pointer - start_frame) % stride == 0:
                frame = LoadImage.convert_to_img(frame)
                frame_list.append(frame)
        capture.release()

        video_data = self.model.vid_trans(frame_list)

        return {'video_data': video_data, 'y': y}

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        video = self.model(input)
        return {'video': video}

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        video = tensor2vid(inputs['video'], self.model.cfg.mean,
                           self.model.cfg.std)
        output_video_path = post_params.get('output_video', None)
        temp_video_file = False
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
            temp_video_file = True

        temp_dir = tempfile.mkdtemp()
        for fid, frame in enumerate(video):
            tpth = os.path.join(temp_dir, '%06d.png' % (fid + 1))
            cv2.imwrite(tpth, frame[:, :, ::-1],
                        [int(cv2.IMWRITE_JPEG_QUALITY), 100])

        cmd = f'ffmpeg -y -f image2 -loglevel quiet -framerate 8.0 -i {temp_dir}/%06d.png \
        -vcodec libx264 -crf 17 -pix_fmt yuv420p {output_video_path}'

        status, output = subprocess.getstatusoutput(cmd)
        if status != 0:
            logger.error('Save Video Error with {}'.format(output))
        os.system(f'rm -rf {temp_dir}')

        if temp_video_file:
            video_file_content = b''
            with open(output_video_path, 'rb') as f:
                video_file_content = f.read()
            os.remove(output_video_path)
            return {OutputKeys.OUTPUT_VIDEO: video_file_content}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}


def tensor2vid(video, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean = torch.tensor(mean, device=video.device).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(std, device=video.device).reshape(1, -1, 1, 1, 1)

    video = video.mul_(std).add_(mean)
    video.clamp_(0, 1)
    video = video * 255.0

    images = rearrange(video, 'b c f h w -> b f h w c')[0]
    images = [(img.numpy()).astype('uint8') for img in images]

    return images
