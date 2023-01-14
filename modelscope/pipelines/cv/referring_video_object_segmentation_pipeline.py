# The implementation here is modified based on MTTR,
# originally Apache 2.0 License and publicly avaialbe at https://github.com/mttr2021/MTTR
# Copyright (c) Alibaba, Inc. and its affiliates.

import tempfile
from typing import Any, Dict

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from einops import rearrange
from moviepy.editor import AudioFileClip, ImageSequenceClip, VideoFileClip
from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.referring_video_object_segmentation,
    module_name=Pipelines.referring_video_object_segmentation)
class ReferringVideoObjectSegmentationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """use `model` to create a referring video object segmentation pipeline for prediction

        Args:
            model: model id on modelscope hub
            render: whether to generate output video for demo service, default: False
        """
        _device = kwargs.pop('device', 'gpu')
        if torch.cuda.is_available() and _device == 'gpu':
            self.device = 'gpu'
        else:
            self.device = 'cpu'
        super().__init__(model=model, device=self.device, **kwargs)

        logger.info('Load model done!')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        """

        Args:
            input: path of the input video

        """
        assert isinstance(input, tuple) and len(
            input
        ) == 2, 'error - input type must be tuple and input length must be 2'
        self.input_video_pth, text_queries = input

        assert 1 <= len(
            text_queries) <= 2, 'error - 1-2 input text queries are expected'

        # extract the relevant subclip:
        self.input_clip_pth = 'input_clip.mp4'

        with VideoFileClip(self.input_video_pth) as video:
            subclip = video.subclip()
            subclip.write_videofile(self.input_clip_pth)

        self.window_length = 24  # length of window during inference
        self.window_overlap = 6  # overlap (in frames) between consecutive windows

        self.video, audio, self.meta = torchvision.io.read_video(
            filename=self.input_clip_pth)
        self.video = rearrange(self.video, 't h w c -> t c h w')

        input_video = F.resize(self.video, size=360, max_size=640)
        if self.device_name == 'gpu':
            input_video = input_video.cuda()

        input_video = input_video.to(torch.float).div_(255)
        input_video = F.normalize(
            input_video, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        video_metadata = {
            'resized_frame_size': input_video.shape[-2:],
            'original_frame_size': self.video.shape[-2:]
        }

        # partition the clip into overlapping windows of frames:
        windows = [
            input_video[i:i + self.window_length]
            for i in range(0, len(input_video), self.window_length
                           - self.window_overlap)
        ]
        # clean up the text queries:
        self.text_queries = [' '.join(q.lower().split()) for q in text_queries]

        result = {
            'text_queries': self.text_queries,
            'windows': windows,
            'video_metadata': video_metadata
        }

        return result

    def forward(self, input: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        with torch.no_grad():
            pred_masks_per_query = []
            t, _, h, w = self.video.shape
            for text_query in tqdm(input['text_queries'], desc='text queries'):
                pred_masks = torch.zeros(size=(t, 1, h, w))
                for i, window in enumerate(
                        tqdm(input['windows'], desc='windows')):

                    window_masks = self.model.inference(
                        window=window,
                        text_query=text_query,
                        metadata=input['video_metadata'])

                    win_start_idx = i * (
                        self.window_length - self.window_overlap)
                    pred_masks[win_start_idx:win_start_idx
                               + self.window_length] = window_masks
                pred_masks_per_query.append(pred_masks)
        return pred_masks_per_query

    def postprocess(self, inputs, **kwargs) -> Dict[str, Any]:
        output_clip_path = None
        render = kwargs.get('render', False)
        if render:
            self.model.cfg.pipeline.save_masked_video = True
        if self.model.cfg.pipeline.save_masked_video:
            # RGB colors for instance masks:
            light_blue = (41, 171, 226)
            purple = (237, 30, 121)
            dark_green = (35, 161, 90)
            orange = (255, 148, 59)
            colors = np.array([light_blue, purple, dark_green, orange])

            # width (in pixels) of the black strip above the video on which the text queries will be displayed:
            text_border_height_per_query = 36

            video_np = rearrange(self.video,
                                 't c h w -> t h w c').numpy() / 255.0

            # set font for text query in output video
            if self.model.cfg.pipeline.output_font:
                try:
                    font = ImageFont.truetype(
                        font=self.model.cfg.pipeline.output_font,
                        size=self.model.cfg.pipeline.output_font_size)
                except OSError:
                    logger.error('can\'t open resource %s, load default font'
                                 % self.model.cfg.pipeline.output_font)
                    font = ImageFont.load_default()
            else:
                font = ImageFont.load_default()

            # del video
            pred_masks_per_frame = rearrange(
                torch.stack(inputs), 'q t 1 h w -> t q h w').numpy()
            masked_video = []
            for vid_frame, frame_masks in tqdm(
                    zip(video_np, pred_masks_per_frame),
                    total=len(video_np),
                    desc='applying masks...'):
                # apply the masks:
                for inst_mask, color in zip(frame_masks, colors):
                    vid_frame = apply_mask(vid_frame, inst_mask, color / 255.0)
                vid_frame = Image.fromarray((vid_frame * 255).astype(np.uint8))
                # visualize the text queries:
                vid_frame = ImageOps.expand(
                    vid_frame,
                    border=(0, len(self.text_queries)
                            * text_border_height_per_query, 0, 0))
                W, H = vid_frame.size
                draw = ImageDraw.Draw(vid_frame)

                for i, (text_query, color) in enumerate(
                        zip(self.text_queries, colors), start=1):
                    w, h = draw.textsize(text_query, font=font)
                    draw.text(((W - w) / 2,
                               (text_border_height_per_query * i) - h - 3),
                              text_query,
                              fill=tuple(color) + (255, ),
                              font=font)
                masked_video.append(np.array(vid_frame))
            # generate and save the output clip:

            output_clip_path = self.model.cfg.pipeline.get(
                'output_path',
                tempfile.NamedTemporaryFile(suffix='.mp4').name)
            clip = ImageSequenceClip(
                sequence=masked_video, fps=self.meta['video_fps'])

            audio_flag = True
            try:
                audio = AudioFileClip(self.input_clip_pth)
            except KeyError as e:
                logger.error(f'key error: {e}!')
                audio_flag = False

            if audio_flag:
                clip = clip.set_audio(audio)
            clip.write_videofile(
                output_clip_path, fps=self.meta['video_fps'], audio=audio_flag)
            del masked_video

        masks = [mask.squeeze(1).cpu().numpy() for mask in inputs]

        fps = self.meta['video_fps']
        output_timestamps = []
        for frame_idx in range(self.video.shape[0]):
            output_timestamps.append(timestamp_format(seconds=frame_idx / fps))
        result = {
            OutputKeys.MASKS: None if render else masks,
            OutputKeys.TIMESTAMPS: None if render else output_timestamps,
            OutputKeys.OUTPUT_VIDEO: output_clip_path
        }

        return result


def apply_mask(image, mask, color, transparency=0.7):
    mask = mask[..., np.newaxis].repeat(repeats=3, axis=2)
    mask = mask * transparency
    color_matrix = np.ones(image.shape, dtype=np.float) * color
    out_image = color_matrix * mask + image * (1.0 - mask)
    return out_image


def timestamp_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = '%02d:%02d:%06.3f' % (h, m, s)
    return time
