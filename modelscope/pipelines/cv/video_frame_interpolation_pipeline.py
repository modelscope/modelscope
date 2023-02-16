# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import math
import os
import os.path as osp
import subprocess
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_frame_interpolation.utils.scene_change_detection import \
    do_scene_detect
from modelscope.models.cv.video_frame_interpolation.VFINet_for_video_frame_interpolation import \
    VFINetForVideoFrameInterpolation
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.preprocessors.cv import VideoReader
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

VIDEO_EXTENSIONS = ('.mp4', '.mov')
logger = get_logger()


def img_trans(img_tensor):  # in format of RGB
    img_tensor = img_tensor / 255.0
    mean = torch.Tensor([0.429, 0.431, 0.397]).view(1, 3, 1,
                                                    1).type_as(img_tensor)
    img_tensor -= mean
    return img_tensor


def add_mean(x):
    mean = torch.Tensor([0.429, 0.431, 0.397]).view(1, 3, 1, 1).type_as(x)
    return x + mean


def img_padding(img_tensor, height, width, pad_num=32):
    ph = ((height - 1) // pad_num + 1) * pad_num
    pw = ((width - 1) // pad_num + 1) * pad_num
    padding = (0, pw - width, 0, ph - height)
    img_tensor = F.pad(img_tensor, padding)
    return img_tensor


def do_inference_lowers(flow_10,
                        flow_12,
                        flow_21,
                        flow_23,
                        img1,
                        img2,
                        inter_model,
                        read_count,
                        inter_count,
                        delta,
                        outputs,
                        start_end_flag=False):
    # given frame1, frame2 and optical flow, predict frame_t
    if start_end_flag:
        read_count -= 1
    else:
        read_count -= 2
    while inter_count <= read_count:
        t = inter_count + 1 - read_count
        t = round(t, 2)
        if (t - 0) < delta / 2:
            output = img1
        elif (1 - t) < delta / 2:
            output = img2
        else:
            output = inter_model(flow_10, flow_12, flow_21, flow_23, img1,
                                 img2, t)

        output = 255 * add_mean(output)
        outputs.append(output)
        inter_count += delta

    return outputs, inter_count


def do_inference_highers(flow_10,
                         flow_12,
                         flow_21,
                         flow_23,
                         img1,
                         img2,
                         img1_up,
                         img2_up,
                         inter_model,
                         read_count,
                         inter_count,
                         delta,
                         outputs,
                         start_end_flag=False):
    # given frame1, frame2 and optical flow, predict frame_t. For videos with a resolution of 2k and above
    if start_end_flag:
        read_count -= 1
    else:
        read_count -= 2
    while inter_count <= read_count:
        t = inter_count + 1 - read_count
        t = round(t, 2)
        if (t - 0) < delta / 2:
            output = img1_up
        elif (1 - t) < delta / 2:
            output = img2_up
        else:
            output = inter_model(flow_10, flow_12, flow_21, flow_23, img1,
                                 img2, img1_up, img2_up, t)

        output = 255 * add_mean(output)
        outputs.append(output)
        inter_count += delta

    return outputs, inter_count


def inference_lowers(flow_model, refine_model, inter_model, video_len,
                     read_count, inter_count, delta, scene_change_flag,
                     img_tensor_list, img_ori_list, inputs, outputs):
    # given a video with a resolution less than 2k and output fps, execute the video frame interpolation function.
    height, width = inputs[read_count].size(2), inputs[read_count].size(3)
    # We use four consecutive frames to do frame interpolation. flow_10 represents
    # optical flow from frame0 to frame1. The similar goes for flow_12, flow_21 and
    # flow_23.
    flow_10 = None
    flow_12 = None
    flow_21 = None
    flow_23 = None
    with torch.no_grad():
        while (read_count < video_len):
            img = inputs[read_count]
            img = img_padding(img, height, width)
            img_ori_list.append(img)
            img_tensor_list.append(img_trans(img))
            read_count += 1
            if len(img_tensor_list) == 2:
                img0 = img_tensor_list[0]
                img1 = img_tensor_list[1]
                img0_ori = img_ori_list[0]
                img1_ori = img_ori_list[1]
                _, flow_01_up = flow_model(
                    img0_ori, img1_ori, iters=12, test_mode=True)
                _, flow_10_up = flow_model(
                    img1_ori, img0_ori, iters=12, test_mode=True)
                flow_01, flow_10 = refine_model(img0, img1, flow_01_up,
                                                flow_10_up, 2)
                scene_change_flag[0] = do_scene_detect(
                    flow_01[:, :, 0:height, 0:width], flow_10[:, :, 0:height,
                                                              0:width],
                    img_ori_list[0][:, :, 0:height, 0:width],
                    img_ori_list[1][:, :, 0:height, 0:width])
                if scene_change_flag[0]:
                    outputs, inter_count = do_inference_lowers(
                        None,
                        None,
                        None,
                        None,
                        img0,
                        img1,
                        inter_model,
                        read_count,
                        inter_count,
                        delta,
                        outputs,
                        start_end_flag=True)
                else:
                    outputs, inter_count = do_inference_lowers(
                        None,
                        flow_01,
                        flow_10,
                        None,
                        img0,
                        img1,
                        inter_model,
                        read_count,
                        inter_count,
                        delta,
                        outputs,
                        start_end_flag=True)

            if len(img_tensor_list) == 4:
                if flow_12 is None or flow_21 is None:
                    img2 = img_tensor_list[2]
                    img2_ori = img_ori_list[2]
                    _, flow_12_up = flow_model(
                        img1_ori, img2_ori, iters=12, test_mode=True)
                    _, flow_21_up = flow_model(
                        img2_ori, img1_ori, iters=12, test_mode=True)
                    flow_12, flow_21 = refine_model(img1, img2, flow_12_up,
                                                    flow_21_up, 2)
                    scene_change_flag[1] = do_scene_detect(
                        flow_12[:, :, 0:height,
                                0:width], flow_21[:, :, 0:height, 0:width],
                        img_ori_list[1][:, :, 0:height, 0:width],
                        img_ori_list[2][:, :, 0:height, 0:width])

                img3 = img_tensor_list[3]
                img3_ori = img_ori_list[3]
                _, flow_23_up = flow_model(
                    img2_ori, img3_ori, iters=12, test_mode=True)
                _, flow_32_up = flow_model(
                    img3_ori, img2_ori, iters=12, test_mode=True)
                flow_23, flow_32 = refine_model(img2, img3, flow_23_up,
                                                flow_32_up, 2)
                scene_change_flag[2] = do_scene_detect(
                    flow_23[:, :, 0:height, 0:width], flow_32[:, :, 0:height,
                                                              0:width],
                    img_ori_list[2][:, :, 0:height, 0:width],
                    img_ori_list[3][:, :, 0:height, 0:width])

                if scene_change_flag[1]:
                    outputs, inter_count = do_inference_lowers(
                        None, None, None, None, img1, img2, inter_model,
                        read_count, inter_count, delta, outputs)
                elif scene_change_flag[0] or scene_change_flag[2]:
                    outputs, inter_count = do_inference_lowers(
                        None, flow_12, flow_21, None, img1, img2, inter_model,
                        read_count, inter_count, delta, outputs)
                else:
                    outputs, inter_count = do_inference_lowers(
                        flow_10_up, flow_12, flow_21, flow_23_up, img1, img2,
                        inter_model, read_count, inter_count, delta, outputs)

                img_tensor_list.pop(0)
                img_ori_list.pop(0)

                # for next group
                img1 = img2
                img2 = img3
                img1_ori = img2_ori
                img2_ori = img3_ori
                flow_10 = flow_21
                flow_12 = flow_23
                flow_21 = flow_32

                flow_10_up = flow_21_up
                flow_12_up = flow_23_up
                flow_21_up = flow_32_up

                # save scene change flag for next group
                scene_change_flag[0] = scene_change_flag[1]
                scene_change_flag[1] = scene_change_flag[2]
                scene_change_flag[2] = False

        if read_count > 0:  # the last remaining 3 images
            img_ori_list.pop(0)
            img_tensor_list.pop(0)
            assert (len(img_tensor_list) == 2)

            if scene_change_flag[1]:
                outputs, inter_count = do_inference_lowers(
                    None,
                    None,
                    None,
                    None,
                    img1,
                    img2,
                    inter_model,
                    read_count,
                    inter_count,
                    delta,
                    outputs,
                    start_end_flag=True)
            else:
                outputs, inter_count = do_inference_lowers(
                    None,
                    flow_12,
                    flow_21,
                    None,
                    img1,
                    img2,
                    inter_model,
                    read_count,
                    inter_count,
                    delta,
                    outputs,
                    start_end_flag=True)

    return outputs


def inference_highers(flow_model, refine_model, inter_model, video_len,
                      read_count, inter_count, delta, scene_change_flag,
                      img_tensor_list, img_ori_list, inputs, outputs):
    # given a video with a resolution of 2k or above and output fps, execute the video frame interpolation function.
    if inputs[read_count].size(2) % 2 != 0 or inputs[read_count].size(
            3) % 2 != 0:
        raise RuntimeError('Video width and height must be even')

    height, width = inputs[read_count].size(2) // 2, inputs[read_count].size(
        3) // 2
    # We use four consecutive frames to do frame interpolation. flow_10 represents
    # optical flow from frame0 to frame1. The similar goes for flow_12, flow_21 and
    # flow_23.
    flow_10 = None
    flow_12 = None
    flow_21 = None
    flow_23 = None
    img_up_list = []
    with torch.no_grad():
        while (read_count < video_len):
            img_up = inputs[read_count]
            img_up = img_padding(img_up, height * 2, width * 2, pad_num=64)
            img = F.interpolate(
                img_up, scale_factor=0.5, mode='bilinear', align_corners=False)

            img_up_list.append(img_trans(img_up))
            img_ori_list.append(img)
            img_tensor_list.append(img_trans(img))
            read_count += 1
            if len(img_tensor_list) == 2:
                img0 = img_tensor_list[0]
                img1 = img_tensor_list[1]
                img0_ori = img_ori_list[0]
                img1_ori = img_ori_list[1]
                img0_up = img_up_list[0]
                img1_up = img_up_list[1]
                _, flow_01_up = flow_model(
                    img0_ori, img1_ori, iters=12, test_mode=True)
                _, flow_10_up = flow_model(
                    img1_ori, img0_ori, iters=12, test_mode=True)
                flow_01, flow_10 = refine_model(img0, img1, flow_01_up,
                                                flow_10_up, 2)
                scene_change_flag[0] = do_scene_detect(
                    flow_01[:, :, 0:height, 0:width], flow_10[:, :, 0:height,
                                                              0:width],
                    img_ori_list[0][:, :, 0:height, 0:width],
                    img_ori_list[1][:, :, 0:height, 0:width])
                if scene_change_flag[0]:
                    outputs, inter_count = do_inference_highers(
                        None,
                        None,
                        None,
                        None,
                        img0,
                        img1,
                        img0_up,
                        img1_up,
                        inter_model,
                        read_count,
                        inter_count,
                        delta,
                        outputs,
                        start_end_flag=True)
                else:
                    outputs, inter_count = do_inference_highers(
                        None,
                        flow_01,
                        flow_10,
                        None,
                        img0,
                        img1,
                        img0_up,
                        img1_up,
                        inter_model,
                        read_count,
                        inter_count,
                        delta,
                        outputs,
                        start_end_flag=True)

            if len(img_tensor_list) == 4:
                if flow_12 is None or flow_21 is None:
                    img2 = img_tensor_list[2]
                    img2_ori = img_ori_list[2]
                    img2_up = img_up_list[2]
                    _, flow_12_up = flow_model(
                        img1_ori, img2_ori, iters=12, test_mode=True)
                    _, flow_21_up = flow_model(
                        img2_ori, img1_ori, iters=12, test_mode=True)
                    flow_12, flow_21 = refine_model(img1, img2, flow_12_up,
                                                    flow_21_up, 2)
                    scene_change_flag[1] = do_scene_detect(
                        flow_12[:, :, 0:height,
                                0:width], flow_21[:, :, 0:height, 0:width],
                        img_ori_list[1][:, :, 0:height, 0:width],
                        img_ori_list[2][:, :, 0:height, 0:width])

                img3 = img_tensor_list[3]
                img3_ori = img_ori_list[3]
                img3_up = img_up_list[3]
                _, flow_23_up = flow_model(
                    img2_ori, img3_ori, iters=12, test_mode=True)
                _, flow_32_up = flow_model(
                    img3_ori, img2_ori, iters=12, test_mode=True)
                flow_23, flow_32 = refine_model(img2, img3, flow_23_up,
                                                flow_32_up, 2)
                scene_change_flag[2] = do_scene_detect(
                    flow_23[:, :, 0:height, 0:width], flow_32[:, :, 0:height,
                                                              0:width],
                    img_ori_list[2][:, :, 0:height, 0:width],
                    img_ori_list[3][:, :, 0:height, 0:width])

                if scene_change_flag[1]:
                    outputs, inter_count = do_inference_highers(
                        None, None, None, None, img1, img2, img1_up, img2_up,
                        inter_model, read_count, inter_count, delta, outputs)
                elif scene_change_flag[0] or scene_change_flag[2]:
                    outputs, inter_count = do_inference_highers(
                        None, flow_12, flow_21, None, img1, img2, img1_up,
                        img2_up, inter_model, read_count, inter_count, delta,
                        outputs)
                else:
                    outputs, inter_count = do_inference_highers(
                        flow_10_up, flow_12, flow_21, flow_23_up, img1, img2,
                        img1_up, img2_up, inter_model, read_count, inter_count,
                        delta, outputs)

                img_up_list.pop(0)
                img_tensor_list.pop(0)
                img_ori_list.pop(0)

                # for next group
                img1 = img2
                img2 = img3
                img1_ori = img2_ori
                img2_ori = img3_ori
                img1_up = img2_up
                img2_up = img3_up
                flow_10 = flow_21
                flow_12 = flow_23
                flow_21 = flow_32

                flow_10_up = flow_21_up
                flow_12_up = flow_23_up
                flow_21_up = flow_32_up

                # save scene change flag for next group
                scene_change_flag[0] = scene_change_flag[1]
                scene_change_flag[1] = scene_change_flag[2]
                scene_change_flag[2] = False

        if read_count > 0:  # the last remaining 3 images
            img_ori_list.pop(0)
            img_tensor_list.pop(0)
            assert (len(img_tensor_list) == 2)

            if scene_change_flag[1]:
                outputs, inter_count = do_inference_highers(
                    None,
                    None,
                    None,
                    None,
                    img1,
                    img2,
                    img1_up,
                    img2_up,
                    inter_model,
                    read_count,
                    inter_count,
                    delta,
                    outputs,
                    start_end_flag=True)
            else:
                outputs, inter_count = do_inference_highers(
                    None,
                    flow_12,
                    flow_21,
                    None,
                    img1,
                    img2,
                    img1_up,
                    img2_up,
                    inter_model,
                    read_count,
                    inter_count,
                    delta,
                    outputs,
                    start_end_flag=True)

    return outputs


def convert(param):
    return {
        k.replace('module.', ''): v
        for k, v in param.items() if 'module.' in k
    }


__all__ = ['VideoFrameInterpolationPipeline']


@PIPELINES.register_module(
    Tasks.video_frame_interpolation,
    module_name=Pipelines.video_frame_interpolation)
class VideoFrameInterpolationPipeline(Pipeline):
    """ Video Frame Interpolation Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> from modelscope.outputs import OutputKeys

    >>> video = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/videos/video_frame_interpolation_test.mp4'
    >>> video_frame_interpolation_pipeline = pipeline(Tasks.video_frame_interpolation,
    'damo/cv_raft_video-frame-interpolation')
    >>> result = video_frame_interpolation_pipeline(video)[OutputKeys.OUTPUT_VIDEO]
    >>> print('pipeline: the output video path is {}'.format(result))
    """

    def __init__(self,
                 model: Union[VFINetForVideoFrameInterpolation, str],
                 preprocessor=None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.net = self.model.model
        self.net.to(self._device)
        self.net.eval()
        logger.info('load video frame-interpolation done')

    def preprocess(self, input: Input, out_fps: float = 0) -> Dict[str, Any]:
        # Determine the input type
        if isinstance(input, str):
            video_reader = VideoReader(input)
        elif isinstance(input, dict):
            video_reader = VideoReader(input['video'])
        inputs = []
        for frame in video_reader:
            inputs.append(frame)
        fps = video_reader.fps

        for i, img in enumerate(inputs):
            img = torch.from_numpy(img.copy()).permute(2, 0, 1).float()
            inputs[i] = img.unsqueeze(0)

        if isinstance(input, str):
            out_fps = 2 * fps
        elif isinstance(input, dict):
            if 'interp_ratio' in input:
                out_fps = input['interp_ratio'] * fps
            elif 'out_fps' in input:
                out_fps = input['out_fps']
            else:
                out_fps = 2 * fps
        return {'video': inputs, 'fps': fps, 'out_fps': out_fps}

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        inputs = input['video']
        fps = input['fps']
        out_fps = input['out_fps']
        video_len = len(inputs)

        flow_model = self.net.flownet
        refine_model = self.net.internet.ifnet

        read_count = 0
        inter_count = 0
        delta = fps / out_fps
        scene_change_flag = [False, False, False]
        img_tensor_list = []
        img_ori_list = []
        outputs = []
        height, width = inputs[read_count].size(2), inputs[read_count].size(3)
        if height >= 1440 or width >= 2560:
            inter_model = self.net.internet_Ds.internet
            outputs = inference_highers(flow_model, refine_model, inter_model,
                                        video_len, read_count, inter_count,
                                        delta, scene_change_flag,
                                        img_tensor_list, img_ori_list, inputs,
                                        outputs)
        else:
            inter_model = self.net.internet.internet
            outputs = inference_lowers(flow_model, refine_model, inter_model,
                                       video_len, read_count, inter_count,
                                       delta, scene_change_flag,
                                       img_tensor_list, img_ori_list, inputs,
                                       outputs)

        for i in range(len(outputs)):
            outputs[i] = outputs[i][:, :, 0:height, 0:width]
        return {'output': outputs, 'fps': out_fps}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_video_path = kwargs.get('output_video', None)
        demo_service = kwargs.get('demo_service', True)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name
        h, w = inputs['output'][0].shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc,
                                       inputs['fps'], (w, h))
        for i in range(len(inputs['output'])):
            img = inputs['output'][i]
            img = img[0].permute(1, 2, 0).byte().cpu().numpy()
            video_writer.write(img.astype(np.uint8))

        video_writer.release()
        if demo_service:
            assert os.system(
                'ffmpeg -version') == 0, 'ffmpeg is not installed correctly!'
            output_video_path_for_web = output_video_path[:-4] + '_web.mp4'
            convert_cmd = f'ffmpeg -i {output_video_path} -vcodec h264 -crf 5 {output_video_path_for_web}'
            subprocess.call(convert_cmd, shell=True)
            return {OutputKeys.OUTPUT_VIDEO: output_video_path_for_web}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}
