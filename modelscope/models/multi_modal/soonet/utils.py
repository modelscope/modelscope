# Copyright 2022-2023 The Alibaba Fundamental Vision  Team Authors. All rights reserved.

import copy

import decord
import numpy as np
from decord import VideoReader, cpu
from decord._ffi.base import DECORDError
from tqdm import tqdm


def decode_video(video_path, target_fps=5):
    """
        Decode video from 'video_path' and return the sampled frames based on target_fps.
        The default value of target_fps is 5.

        Args:
            video_path: the absolute path of video.
            target_fps: the number of sampled video frames per second.

        Returns:
            [imgs, duration]
    """
    decord.bridge.set_bridge('torch')
    vr = VideoReader(video_path, ctx=cpu(0))
    cur_fps = vr.get_avg_fps()
    if cur_fps > target_fps:
        interval = float(cur_fps) / float(target_fps)
        start = float(interval) / 2.
    else:
        interval = 1.0
        start = 0.0

    vid_length = len(vr)
    duration = vid_length / cur_fps
    sampled_idxs = np.clip(
        np.round(np.arange(start, float(vid_length), step=interval)), 0,
        vid_length - 1).astype(np.int32)

    imgs = list()
    for i in tqdm(sampled_idxs):
        bias = 0
        # avoid broken frames
        while bias <= 10:
            try:
                img = vr[i - bias]
                break
            except DECORDError:
                bias += 1
        if bias > 10:
            img = copy.deepcopy(imgs[-1])
            imgs.append(img)
        else:
            img = img / 255.
            img = img.permute(2, 0, 1)
            imgs.append(img)

    return imgs, duration
