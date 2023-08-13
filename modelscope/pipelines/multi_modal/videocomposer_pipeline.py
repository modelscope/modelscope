# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import subprocess
import tempfile
import time
from functools import partial
from typing import Any, Dict

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
from mvextractor.videocap import VideoCap
from PIL import Image

import modelscope.models.multi_modal.videocomposer.data as data
from modelscope.metainfo import Pipelines
from modelscope.models.multi_modal.videocomposer.data.transforms import (
    CenterCropV3, random_resize)
from modelscope.models.multi_modal.videocomposer.ops.random_mask import (
    make_irregular_mask, make_rectangle_mask, make_uncrop)
from modelscope.models.multi_modal.videocomposer.utils.utils import rand_name
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.constant import Tasks
from modelscope.utils.device import device_placement
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.text_to_video_synthesis, module_name=Pipelines.videocomposer)
class VideoComposerPipeline(Pipeline):
    r""" Video Composer Pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks
    >>> pipe = pipeline(
            task=Tasks.text_to_video_synthesis,
            model='buptwq/videocomposer',
            model_revision='v1.0.1')
    >>> inputs = {'Video:FILE': 'path/input_video.mp4',
                  'Image:FILE': 'path/input_image.png',
                  'text': 'the text description'}
    >>> output = pipe(inputs)
    """

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a videocomposer pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model)
        self.log_dir = kwargs.pop('log_dir', './video_outputs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.feature_framerate = kwargs.pop('feature_framerate', 4)
        self.frame_lens = kwargs.pop('frame_lens', [
            16,
            16,
            16,
            16,
        ])
        self.feature_framerates = kwargs.pop('feature_framerates', [
            4,
        ])
        self.batch_sizes = kwargs.pop('batch_sizes', {
            '1': 1,
            '4': 1,
            '8': 1,
            '16': 1,
        })
        l1 = len(self.frame_lens)
        l2 = len(self.feature_framerates)
        self.max_frames = self.frame_lens[0 % (l1 * l2) // l2]
        self.batch_size = self.batch_sizes[str(self.max_frames)]
        self.resolution = kwargs.pop('resolution', 256)
        self.image_resolution = kwargs.pop('image_resolution', 256)
        self.mean = kwargs.pop('mean', [0.5, 0.5, 0.5])
        self.std = kwargs.pop('std', [0.5, 0.5, 0.5])
        self.vit_image_size = kwargs.pop('vit_image_size', 224)
        self.vit_mean = kwargs.pop('vit_mean',
                                   [0.48145466, 0.4578275, 0.40821073])
        self.vit_std = kwargs.pop('vit_std',
                                  [0.26862954, 0.26130258, 0.27577711])
        self.misc_size = kwargs.pop('kwargs.pop', 384)
        self.visual_mv = kwargs.pop('visual_mv', False)
        self.max_words = kwargs.pop('max_words', 1000)
        self.mvs_visual = kwargs.pop('mvs_visual', False)

        self.infer_trans = data.Compose([
            data.CenterCropV2(size=self.resolution),
            data.ToTensor(),
            data.Normalize(mean=self.mean, std=self.std)
        ])

        self.misc_transforms = data.Compose([
            T.Lambda(partial(random_resize, size=self.misc_size)),
            data.CenterCropV2(self.misc_size),
            data.ToTensor()
        ])

        self.mv_transforms = data.Compose(
            [T.Resize(size=self.resolution),
             T.CenterCrop(self.resolution)])

        self.vit_transforms = T.Compose([
            CenterCropV3(self.vit_image_size),
            T.ToTensor(),
            T.Normalize(mean=self.vit_mean, std=self.vit_std)
        ])

    def preprocess(self, input: Input) -> Dict[str, Any]:
        video_key = input['Video:FILE']
        cap_txt = input['text']
        style_image = input['Image:FILE']

        total_frames = None

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            try:
                ref_frame, vit_image, video_data, misc_data, mv_data = self.video_data_preprocess(
                    video_key, self.feature_framerate, total_frames,
                    self.mvs_visual)
            except Exception as e:
                logger.info(
                    '{} get frames failed... with error: {}'.format(
                        video_key, e),
                    flush=True)

                ref_frame = torch.zeros(3, self.vit_image_size,
                                        self.vit_image_size)
                video_data = torch.zeros(self.max_frames, 3,
                                         self.image_resolution,
                                         self.image_resolution)
                misc_data = torch.zeros(self.max_frames, 3, self.misc_size,
                                        self.misc_size)

                mv_data = torch.zeros(self.max_frames, 2,
                                      self.image_resolution,
                                      self.image_resolution)
        else:
            logger.info(
                'The video path does not exist or no video dir provided!')
            ref_frame = torch.zeros(3, self.vit_image_size,
                                    self.vit_image_size)
            _ = torch.zeros(3, self.vit_image_size, self.vit_image_size)
            video_data = torch.zeros(self.max_frames, 3, self.image_resolution,
                                     self.image_resolution)
            misc_data = torch.zeros(self.max_frames, 3, self.misc_size,
                                    self.misc_size)
            mv_data = torch.zeros(self.max_frames, 2, self.image_resolution,
                                  self.image_resolution)

        # inpainting mask
        p = random.random()
        if p < 0.7:
            mask = make_irregular_mask(512, 512)
        elif p < 0.9:
            mask = make_rectangle_mask(512, 512)
        else:
            mask = make_uncrop(512, 512)
        mask = torch.from_numpy(
            cv2.resize(
                mask, (self.misc_size, self.misc_size),
                interpolation=cv2.INTER_NEAREST)).unsqueeze(0).float()

        mask = mask.unsqueeze(0).repeat_interleave(
            repeats=self.max_frames, dim=0)
        video_input = {
            'ref_frame': ref_frame.unsqueeze(0),
            'cap_txt': cap_txt,
            'video_data': video_data.unsqueeze(0),
            'misc_data': misc_data.unsqueeze(0),
            'feature_framerate': feature_framerate,
            'mask': mask.unsqueeze(0),
            'mv_data': mv_data.unsqueeze(0),
            'style_image': style_image
        }
        return video_input

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        return self.model(input)

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        output_video_path = post_params.get('output_video', None)
        temp_video_file = False
        if output_video_path is not None:
            output_video_path = tempfile.NamedTemporaryFile(suffix='.gif').name
            temp_video_file = True

        if temp_video_file:
            return {OutputKeys.OUTPUT_VIDEO: inputs['video_path']}
        else:
            return {OutputKeys.OUTPUT_VIDEO: inputs['video']}

    def video_data_preprocess(self, video_key, feature_framerate, total_frames,
                              visual_mv):

        filename = video_key
        for _ in range(5):
            try:
                frame_types, frames, mvs, mvs_visual = self.extract_motion_vectors(
                    input_video=filename,
                    fps=feature_framerate,
                    visual_mv=visual_mv)
                break
            except Exception as e:
                logger.error(
                    '{} read video frames and motion vectors failed with error: {}'
                    .format(video_key, e),
                    flush=True)

        total_frames = len(frame_types)
        start_indexs = np.where((np.array(frame_types) == 'I') & (
            total_frames - np.arange(total_frames) >= self.max_frames))[0]
        start_index = np.random.choice(start_indexs)
        indices = np.arange(start_index, start_index + self.max_frames)

        # note frames are in BGR mode, need to trans to RGB mode
        frames = [Image.fromarray(frames[i][:, :, ::-1]) for i in indices]
        mvs = [torch.from_numpy(mvs[i].transpose((2, 0, 1))) for i in indices]
        mvs = torch.stack(mvs)

        if visual_mv:
            images = [(mvs_visual[i][:, :, ::-1]).astype('uint8')
                      for i in indices]
            path = self.log_dir + '/visual_mv/' + video_key.split(
                '/')[-1] + '.gif'
            if not os.path.exists(self.log_dir + '/visual_mv/'):
                os.makedirs(self.log_dir + '/visual_mv/', exist_ok=True)
            logger.info('save motion vectors visualization to :', path)
            imageio.mimwrite(path, images, fps=8)

        have_frames = len(frames) > 0
        middle_indix = int(len(frames) / 2)
        if have_frames:
            ref_frame = frames[middle_indix]
            vit_image = self.vit_transforms(ref_frame)
            misc_imgs_np = self.misc_transforms[:2](frames)
            misc_imgs = self.misc_transforms[2:](misc_imgs_np)
            frames = self.infer_trans(frames)
            mvs = self.mv_transforms(mvs)
        else:
            vit_image = torch.zeros(3, self.vit_image_size,
                                    self.vit_image_size)

        video_data = torch.zeros(self.max_frames, 3, self.image_resolution,
                                 self.image_resolution)
        mv_data = torch.zeros(self.max_frames, 2, self.image_resolution,
                              self.image_resolution)
        misc_data = torch.zeros(self.max_frames, 3, self.misc_size,
                                self.misc_size)
        if have_frames:
            video_data[:len(frames), ...] = frames
            misc_data[:len(frames), ...] = misc_imgs
            mv_data[:len(frames), ...] = mvs

        ref_frame = vit_image

        del frames
        del misc_imgs
        del mvs

        return ref_frame, vit_image, video_data, misc_data, mv_data

    def extract_motion_vectors(self,
                               input_video,
                               fps=4,
                               dump=False,
                               verbose=False,
                               visual_mv=False):

        if dump:
            now = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            for child in ['frames', 'motion_vectors']:
                os.makedirs(os.path.join(f'out-{now}', child), exist_ok=True)
        temp = rand_name()
        tmp_video = os.path.join(
            input_video.split('/')[0], f'{temp}' + input_video.split('/')[-1])
        videocapture = cv2.VideoCapture(input_video)
        frames_num = videocapture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps_video = videocapture.get(cv2.CAP_PROP_FPS)
        # check if enough frames
        if frames_num / fps_video * fps > 16:
            fps = max(fps, 1)
        else:
            fps = int(16 / (frames_num / fps_video)) + 1
        ffmpeg_cmd = f'ffmpeg -threads 8 -loglevel error -i {input_video} -filter:v \
                        fps={fps} -c:v mpeg4 -f rawvideo {tmp_video}'

        if os.path.exists(tmp_video):
            os.remove(tmp_video)

        subprocess.run(args=ffmpeg_cmd, shell=True, timeout=120)

        cap = VideoCap()
        # open the video file
        ret = cap.open(tmp_video)
        if not ret:
            raise RuntimeError(f'Could not open {tmp_video}')

        step = 0
        times = []

        frame_types = []
        frames = []
        mvs = []
        mvs_visual = []
        # continuously read and display video frames and motion vectors
        while True:
            if verbose:
                logger.info('Frame: ', step, end=' ')

            tstart = time.perf_counter()

            # read next video frame and corresponding motion vectors
            ret, frame, motion_vectors, frame_type, timestamp = cap.read()

            tend = time.perf_counter()
            telapsed = tend - tstart
            times.append(telapsed)

            # if there is an error reading the frame
            if not ret:
                if verbose:
                    logger.warning('No frame read. Stopping.')
                break

            frame_save = np.zeros(frame.copy().shape, dtype=np.uint8)
            if visual_mv:
                frame_save = draw_motion_vectors(frame_save, motion_vectors)

            # store motion vectors, frames, etc. in output directory
            dump = False
            if frame.shape[1] >= frame.shape[0]:
                w_half = (frame.shape[1] - frame.shape[0]) // 2
                if dump:
                    cv2.imwrite(
                        os.path.join('./mv_visual/', f'frame-{step}.jpg'),
                        frame_save[:, w_half:-w_half])
                mvs_visual.append(frame_save[:, w_half:-w_half])
            else:
                h_half = (frame.shape[0] - frame.shape[1]) // 2
                if dump:
                    cv2.imwrite(
                        os.path.join('./mv_visual/', f'frame-{step}.jpg'),
                        frame_save[h_half:-h_half, :])
                mvs_visual.append(frame_save[h_half:-h_half, :])

            h, w = frame.shape[:2]
            mv = np.zeros((h, w, 2))
            position = motion_vectors[:, 5:7].clip((0, 0), (w - 1, h - 1))
            mv[position[:, 1],
               position[:,
                        0]] = motion_vectors[:, 0:
                                             1] * motion_vectors[:, 7:
                                                                 9] / motion_vectors[:,
                                                                                     9:]

            step += 1
            frame_types.append(frame_type)
            frames.append(frame)
            mvs.append(mv)
        if verbose:
            logger.info('average dt: ', np.mean(times))
        cap.release()

        if os.path.exists(tmp_video):
            os.remove(tmp_video)

        return frame_types, frames, mvs, mvs_visual
