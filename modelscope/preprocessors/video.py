import math
import os
import random
import uuid
from os.path import exists
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data
import torch.utils.dlpack as dlpack
import torchvision.transforms._transforms_video as transforms
from decord import VideoReader
from torchvision.transforms import Compose

from modelscope.hub.file_download import http_get_file
from modelscope.metainfo import Preprocessors
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.type_assert import type_assert
from .base import Preprocessor
from .builder import PREPROCESSORS


def ReadVideoData(cfg,
                  video_path,
                  num_spatial_crops_override=None,
                  num_temporal_views_override=None):
    """ simple interface to load video frames from file

    Args:
        cfg (Config): The global config object.
        video_path (str): video file path
        num_spatial_crops_override (int): the spatial crops per clip
        num_temporal_views_override (int): the temporal clips per video
    Returns:
        data (Tensor): the normalized video clips for model inputs
    """
    url_parsed = urlparse(video_path)
    if url_parsed.scheme in ('file', '') and exists(
            url_parsed.path):  # Possibly a local file
        data = _decode_video(cfg, video_path, num_temporal_views_override)
    else:
        with TemporaryDirectory() as temporary_cache_dir:
            random_str = uuid.uuid4().hex
            http_get_file(
                url=video_path,
                local_dir=temporary_cache_dir,
                file_name=random_str,
                cookies=None)
            temp_file_path = os.path.join(temporary_cache_dir, random_str)
            data = _decode_video(cfg, temp_file_path,
                                 num_temporal_views_override)

    if num_spatial_crops_override is not None:
        num_spatial_crops = num_spatial_crops_override
        transform = kinetics400_tranform(cfg, num_spatial_crops_override)
    else:
        num_spatial_crops = cfg.TEST.NUM_SPATIAL_CROPS
        transform = kinetics400_tranform(cfg, cfg.TEST.NUM_SPATIAL_CROPS)
    data_list = []
    for i in range(data.size(0)):
        for j in range(num_spatial_crops):
            transform.transforms[1].set_spatial_index(j)
            data_list.append(transform(data[i]))
    return torch.stack(data_list, dim=0)


def kinetics400_tranform(cfg, num_spatial_crops):
    """
    Configs the transform for the kinetics-400 dataset.
    We apply controlled spatial cropping and normalization.
    Args:
        cfg (Config): The global config object.
        num_spatial_crops (int): the spatial crops per clip
    Returns:
        transform_function (Compose): the transform function for input clips
    """
    resize_video = KineticsResizedCrop(
        short_side_range=[cfg.DATA.TEST_SCALE, cfg.DATA.TEST_SCALE],
        crop_size=cfg.DATA.TEST_CROP_SIZE,
        num_spatial_crops=num_spatial_crops)
    std_transform_list = [
        transforms.ToTensorVideo(), resize_video,
        transforms.NormalizeVideo(
            mean=cfg.DATA.MEAN, std=cfg.DATA.STD, inplace=True)
    ]
    return Compose(std_transform_list)


def _interval_based_sampling(vid_length, vid_fps, target_fps, clip_idx,
                             num_clips, num_frames, interval, minus_interval):
    """
        Generates the frame index list using interval based sampling.

        Args:
            vid_length (int): the length of the whole video (valid selection range).
            vid_fps (int): the original video fps
            target_fps (int): the normalized video fps
            clip_idx (int):
                -1 for random temporal sampling, and positive values for sampling specific
                clip from the video
            num_clips (int):
                the total clips to be sampled from each video. combined with clip_idx,
                the sampled video is the "clip_idx-th" video from "num_clips" videos.
            num_frames (int): number of frames in each sampled clips.
            interval (int): the interval to sample each frame.
            minus_interval (bool): control the end index

        Returns:
            index (tensor): the sampled frame indexes
    """
    if num_frames == 1:
        index = [random.randint(0, vid_length - 1)]
    else:
        # transform FPS
        clip_length = num_frames * interval * vid_fps / target_fps

        max_idx = max(vid_length - clip_length, 0)
        if num_clips == 1:
            start_idx = max_idx / 2
        else:
            start_idx = clip_idx * math.floor(max_idx / (num_clips - 1))
        if minus_interval:
            end_idx = start_idx + clip_length - interval
        else:
            end_idx = start_idx + clip_length - 1

        index = torch.linspace(start_idx, end_idx, num_frames)
        index = torch.clamp(index, 0, vid_length - 1).long()

    return index


def _decode_video_frames_list(cfg,
                              frames_list,
                              vid_fps,
                              num_temporal_views_override=None):
    """
        Decodes the video given the numpy frames.
        Args:
            cfg          (Config): The global config object.
            frames_list  (list):  all frames for a video, the frames should be numpy array.
            vid_fps      (int):  the fps of this video.
            num_temporal_views_override (int): the temporal clips per video
        Returns:
            frames            (Tensor): video tensor data
    """
    assert isinstance(frames_list, list)
    if num_temporal_views_override is not None:
        num_clips_per_video = num_temporal_views_override
    else:
        num_clips_per_video = cfg.TEST.NUM_ENSEMBLE_VIEWS

    frame_list = []
    for clip_idx in range(num_clips_per_video):
        # for each clip in the video,
        # a list is generated before decoding the specified frames from the video
        list_ = _interval_based_sampling(
            len(frames_list),
            vid_fps,
            cfg.DATA.TARGET_FPS,
            clip_idx,
            num_clips_per_video,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.SAMPLING_RATE,
            cfg.DATA.MINUS_INTERVAL,
        )
        frames = None
        frames = torch.from_numpy(
            np.stack([frames_list[index] for index in list_.tolist()], axis=0))
        frame_list.append(frames)
    frames = torch.stack(frame_list)
    del vr
    return frames


def _decode_video(cfg, path, num_temporal_views_override=None):
    """
        Decodes the video given the numpy frames.
        Args:
            cfg          (Config): The global config object.
            path          (str): video file path.
            num_temporal_views_override (int): the temporal clips per video
        Returns:
            frames            (Tensor): video tensor data
    """
    vr = VideoReader(path)
    if num_temporal_views_override is not None:
        num_clips_per_video = num_temporal_views_override
    else:
        num_clips_per_video = cfg.TEST.NUM_ENSEMBLE_VIEWS

    frame_list = []
    for clip_idx in range(num_clips_per_video):
        # for each clip in the video,
        # a list is generated before decoding the specified frames from the video
        list_ = _interval_based_sampling(
            len(vr),
            vr.get_avg_fps(),
            cfg.DATA.TARGET_FPS,
            clip_idx,
            num_clips_per_video,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.SAMPLING_RATE,
            cfg.DATA.MINUS_INTERVAL,
        )
        frames = None
        if path.endswith('.avi'):
            append_list = torch.arange(0, list_[0], 4)
            frames = dlpack.from_dlpack(
                vr.get_batch(torch.cat([append_list,
                                        list_])).to_dlpack()).clone()
            frames = frames[append_list.shape[0]:]
        else:
            frames = dlpack.from_dlpack(
                vr.get_batch(list_).to_dlpack()).clone()
        frame_list.append(frames)
    frames = torch.stack(frame_list)
    del vr
    return frames


class KineticsResizedCrop(object):
    """Perform resize and crop for kinetics-400 dataset
    Args:
        short_side_range (list): The length of short side range. In inference, this shoudle be [256, 256]
        crop_size         (int): The cropped size for frames.
        num_spatial_crops (int): The number of the cropped spatial regions in each video.
    """

    def __init__(
        self,
        short_side_range,
        crop_size,
        num_spatial_crops=1,
    ):
        self.idx = -1
        self.short_side_range = short_side_range
        self.crop_size = int(crop_size)
        self.num_spatial_crops = num_spatial_crops

    def _get_controlled_crop(self, clip):
        """Perform controlled crop for video tensor.
        Args:
            clip (Tensor): the video data, the shape is [T, C, H, W]
        """
        _, _, clip_height, clip_width = clip.shape

        length = self.short_side_range[0]

        if clip_height < clip_width:
            new_clip_height = int(length)
            new_clip_width = int(clip_width / clip_height * new_clip_height)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode='bilinear')
        else:
            new_clip_width = int(length)
            new_clip_height = int(clip_height / clip_width * new_clip_width)
            new_clip = torch.nn.functional.interpolate(
                clip, size=(new_clip_height, new_clip_width), mode='bilinear')
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        if self.num_spatial_crops == 1:
            x = x_max // 2
            y = y_max // 2
        elif self.num_spatial_crops == 3:
            if self.idx == 0:
                if new_clip_width == length:
                    x = x_max // 2
                    y = 0
                elif new_clip_height == length:
                    x = 0
                    y = y_max // 2
            elif self.idx == 1:
                x = x_max // 2
                y = y_max // 2
            elif self.idx == 2:
                if new_clip_width == length:
                    x = x_max // 2
                    y = y_max
                elif new_clip_height == length:
                    x = x_max
                    y = y_max // 2
        return new_clip[:, :, y:y + self.crop_size, x:x + self.crop_size]

    def _get_random_crop(self, clip):
        _, _, clip_height, clip_width = clip.shape

        short_side = min(clip_height, clip_width)
        long_side = max(clip_height, clip_width)
        new_short_side = int(random.uniform(*self.short_side_range))
        new_long_side = int(long_side / short_side * new_short_side)
        if clip_height < clip_width:
            new_clip_height = new_short_side
            new_clip_width = new_long_side
        else:
            new_clip_height = new_long_side
            new_clip_width = new_short_side

        new_clip = torch.nn.functional.interpolate(
            clip, size=(new_clip_height, new_clip_width), mode='bilinear')

        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        x = int(random.uniform(0, x_max))
        y = int(random.uniform(0, y_max))
        return new_clip[:, :, y:y + self.crop_size, x:x + self.crop_size]

    def set_spatial_index(self, idx):
        """Set the spatial cropping index for controlled cropping..
        Args:
            idx (int): the spatial index. The value should be in [0, 1, 2], means [left, center, right], respectively.
        """
        self.idx = idx

    def __call__(self, clip):
        return self._get_controlled_crop(clip)


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.movie_scene_segmentation_preprocessor)
class MovieSceneSegmentationPreprocessor(Preprocessor):

    def __init__(self, *args, **kwargs):
        """
        movie scene segmentation preprocessor
        """
        super().__init__(*args, **kwargs)

        self.is_train = kwargs.pop('is_train', True)
        self.preprocessor_train_cfg = kwargs.pop(ModeKeys.TRAIN, None)
        self.preprocessor_test_cfg = kwargs.pop(ModeKeys.EVAL, None)
        self.num_keyframe = kwargs.pop('num_keyframe', 3)

        from .movie_scene_segmentation import get_transform
        self.train_transform = get_transform(self.preprocessor_train_cfg)
        self.test_transform = get_transform(self.preprocessor_test_cfg)

    def train(self):
        self.is_train = True
        return

    def eval(self):
        self.is_train = False
        return

    @type_assert(object, object)
    def __call__(self, results):
        if self.is_train:
            transforms = self.train_transform
        else:
            transforms = self.test_transform

        results = torch.stack(transforms(results), dim=0)
        results = results.view(-1, self.num_keyframe, 3, 224, 224)
        return results
