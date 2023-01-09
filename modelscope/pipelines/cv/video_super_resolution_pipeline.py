# The implementation here is modified based on RealBasicVSR,
# originally Apache 2.0 License and publicly avaialbe at
# https://github.com/ckkelvinchan/RealBasicVSR/blob/master/inference_realbasicvsr.py
import math
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional, Union

import cv2
import numpy as np
import torch
from torchvision.utils import make_grid

from modelscope.metainfo import Pipelines
from modelscope.models.cv.video_super_resolution import \
    RealBasicVSRNetForVideoSR
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.cv import VideoReader
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

VIDEO_EXTENSIONS = ('.mp4', '.mov')

logger = get_logger()


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.
    After clamping to (min, max), image values will be normalized to [0, 1].
    For different tensor shapes, this function will have different behaviors:
        1. 4D mini-batch Tensor of shape (N x 3/1 x H x W):
            Use `make_grid` to stitch images in the batch dimension, and then
            convert it to numpy array.
        2. 3D Tensor of shape (3/1 x H x W) and 2D Tensor of shape (H x W):
            Directly change to numpy array.
    Note that the image channel in input tensors should be RGB order. This
    function will convert it to cv2 convention, i.e., (H x W x C) with BGR
    order.
    Args:
        tensor (Tensor | list[Tensor]): Input tensors.
        out_type (numpy type): Output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple): min and max values for clamp.
    Returns:
        (Tensor | list[Tensor]): 3D ndarray of shape (H x W x C) or 2D ndarray
        of shape (H x W).
    """
    condition = torch.is_tensor(tensor) or (isinstance(tensor, list) and all(
        torch.is_tensor(t) for t in tensor))
    if not condition:
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        # Squeeze two times so that:
        # 1. (1, 1, h, w) -> (h, w) or
        # 3. (1, 3, h, w) -> (3, h, w) or
        # 2. (n>1, 3/1, h, w) -> (n>1, 3/1, h, w)
        _tensor = _tensor.squeeze(0).squeeze(0)
        _tensor = _tensor.float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise ValueError('Only support 4D, 3D or 2D tensor. '
                             f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    result = result[0] if len(result) == 1 else result
    return result


@PIPELINES.register_module(
    Tasks.video_super_resolution, module_name=Pipelines.video_super_resolution)
class VideoSuperResolutionPipeline(Pipeline):

    def __init__(self,
                 model: Union[RealBasicVSRNetForVideoSR, str],
                 preprocessor=None,
                 **kwargs):
        super().__init__(model=model, preprocessor=preprocessor, **kwargs)
        self.model.eval()
        self.config = self.model.config

        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')

        logger.info('load video super-resolution model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        # input is a video file
        video_reader = VideoReader(input)
        inputs = []
        for frame in video_reader:
            inputs.append(np.flip(frame, axis=2))
        fps = video_reader.fps

        for i, img in enumerate(inputs):
            img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
            inputs[i] = img.unsqueeze(0)
        inputs = torch.stack(inputs, dim=1)
        return {'video': inputs, 'fps': fps}

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        inputs = input['video']
        with torch.no_grad():
            if isinstance(self.config.model.max_seq_len, int):
                outputs = []
                for i in range(0, inputs.size(1),
                               self.config.model.max_seq_len):
                    imgs = inputs[:,
                                  i:i + self.config.model.max_seq_len, :, :, :]
                    imgs = imgs.to(self._device)
                    outputs.append(
                        self.model._inference_forward(imgs)['output'].cpu())
                outputs = torch.cat(outputs, dim=1)
            else:
                inputs = inputs.to(self._device)
                outputs = self.model._inference_forward(inputs)['output'].cpu()
        return {'output': outputs, 'fps': input['fps']}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_video_path = kwargs.get('output_video', None)
        demo_service = kwargs.get('demo_service', True)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

        h, w = inputs['output'].shape[-2:]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc,
                                       inputs['fps'], (w, h))
        for i in range(0, inputs['output'].size(1)):
            img = tensor2img(inputs['output'][:, i, :, :, :])
            video_writer.write(img.astype(np.uint8))
        video_writer.release()

        if demo_service:
            assert os.system(
                'ffmpeg -version'
            ) == 0, 'ffmpeg is not installed correctly, please refer to https://trac.ffmpeg.org/wiki/CompilationGuide.'
            output_video_path_for_web = output_video_path[:-4] + '_web.mp4'
            convert_cmd = f'ffmpeg -i {output_video_path} -vcodec h264 -crf 5 {output_video_path_for_web}'
            subprocess.call(convert_cmd, shell=True)
            return {OutputKeys.OUTPUT_VIDEO: output_video_path_for_web}
        else:
            return {OutputKeys.OUTPUT_VIDEO: output_video_path}
