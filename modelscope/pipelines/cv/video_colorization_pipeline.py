# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import subprocess
import tempfile
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch
from torchvision import models, transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.image_colorization import (DynamicUnetDeep,
                                                     DynamicUnetWide, NormType)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.cv import VideoReader
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.video_colorization, module_name=Pipelines.video_colorization)
class VideoColorizationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a video colorization pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        self.cut = 8
        self.size = 512
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.orig_img = None
        self.model_type = 'stable'
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.denorm = transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

        if self.model_type == 'stable':
            body = models.resnet101(pretrained=True)
            body = torch.nn.Sequential(*list(body.children())[:self.cut])
            self.model = DynamicUnetWide(
                body,
                n_classes=3,
                blur=True,
                blur_final=True,
                self_attention=True,
                y_range=(-3.0, 3.0),
                norm_type=NormType.Spectral,
                last_cross=True,
                bottle=False,
                nf_factor=2,
            ).to(self.device)
        else:
            body = models.resnet34(pretrained=True)
            body = torch.nn.Sequential(*list(body.children())[:self.cut])
            self.model = DynamicUnetDeep(
                body,
                n_classes=3,
                blur=True,
                blur_final=True,
                self_attention=True,
                y_range=(-3.0, 3.0),
                norm_type=NormType.Spectral,
                last_cross=True,
                bottle=False,
                nf_factor=1.5,
            ).to(self.device)

        model_path = f'{model}/{ModelFile.TORCH_MODEL_FILE}'
        self.model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))['model'],
            strict=True)

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        # input is a video file
        video_reader = VideoReader(input)
        inputs = []
        for frame in video_reader:
            inputs.append(frame)
        fps = video_reader.fps

        self.orig_inputs = inputs.copy()
        self.height, self.width = inputs[0].shape[:2]
        if self.width * self.height < 100000:
            self.size = 256

        for i, img in enumerate(inputs):
            img = PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = img.resize((self.size, self.size),
                             resample=PIL.Image.BILINEAR)
            img = self.norm(img).unsqueeze(0)
            inputs[i] = img

        return {'video': inputs, 'fps': fps}

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for i, img in enumerate(inputs['video']):
                img = img.to(self.device)
                out = self.model(img)[0]

                out = self.denorm(out)
                out = out.float().clamp(min=0, max=1)
                out_img = (out.permute(1, 2, 0).flip(2).cpu().numpy()
                           * 255).astype(np.uint8)

                color_np = cv2.resize(out_img, (self.width, self.height))
                orig_np = np.asarray(self.orig_inputs[i])
                color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
                orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)

                hires = np.copy(orig_yuv)
                hires[:, :, 1:3] = color_yuv[:, :, 1:3]
                out_img = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)

                outputs.append(out_img)

        return {'output': outputs, 'fps': inputs['fps']}

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_video_path = kwargs.get('output_video', None)
        demo_service = kwargs.get('demo_service', True)
        if output_video_path is None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.mp4').name

        h, w = inputs['output'][0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc,
                                       inputs['fps'], (w, h))
        for i in range(len(inputs['output'])):
            img = inputs['output'][i]
            video_writer.write(img)
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
