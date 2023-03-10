# ------------------------------------------------------------------------------------
# BaSSL
# Copyright (c) 2021 KakaoBrain. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# Github: https://github.com/kakaobrain/bassl
# ------------------------------------------------------------------------------------

import random

import numpy as np


class InstanceShotSampler:
    """ This is for instance at pre-training stage """

    def __call__(self, center_sid: int, *args, **kwargs):
        return center_sid


class TemporalShotSampler:
    """ This is for temporal at pre-training stage """

    def __init__(self, neighbor_size: int):
        self.N = neighbor_size

    def __call__(self, center_sid: int, total_num_shot: int):
        """ we randomly sample one shot from neighbor shots within local temporal window
        """
        shot_idx = center_sid + np.arange(
            -self.N, self.N + 1
        )  # total number of neighbor shots = 2N+1 (query (1) + neighbors (2*N))
        shot_idx = np.clip(shot_idx, 0,
                           total_num_shot)  # deal with out-of-boundary indices
        shot_idx = random.choice(
            np.unique(np.delete(shot_idx, np.where(shot_idx == center_sid))))
        return shot_idx


class SequenceShotSampler:
    """ This is for bassl or shotcol at pre-training stage """

    def __init__(self, neighbor_size: int, neighbor_interval: int):
        self.interval = neighbor_interval
        self.window_size = neighbor_size * self.interval  # temporal coverage

    def __call__(self,
                 center_sid: int,
                 total_num_shot: int,
                 sparse_method: str = 'edge'):
        """
        Args:
            center_sid: index of center shot
            total_num_shot: last index of shot for given video
            sparse_stride: stride to sample sparse ones from dense sequence
                    for curriculum learning
        """

        dense_shot_idx = center_sid + np.arange(
            -self.window_size, self.window_size + 1,
            self.interval)  # total number of shots = 2*neighbor_size+1

        if dense_shot_idx[0] < 0:
            # if center_sid is near left-side of video, we shift window rightward
            # so that the leftmost index is 0
            dense_shot_idx -= dense_shot_idx[0]
        elif dense_shot_idx[-1] > (total_num_shot - 1):
            # if center_sid is near right-side of video, we shift window leftward
            # so that the rightmost index is total_num_shot - 1
            dense_shot_idx -= dense_shot_idx[-1] - (total_num_shot - 1)

        # to deal with videos that have smaller number of shots than window size
        dense_shot_idx = np.clip(dense_shot_idx, 0, total_num_shot)

        if sparse_method == 'edge':
            # in this case, we use two edge shots as sparse sequence
            sparse_stride = len(dense_shot_idx) - 1
            sparse_idx_to_dense = np.arange(0, len(dense_shot_idx),
                                            sparse_stride)
        elif sparse_method == 'edge+center':
            # in this case, we use two edge shots + center shot as sparse sequence
            sparse_idx_to_dense = np.array(
                [0, len(dense_shot_idx) - 1,
                 len(dense_shot_idx) // 2])

        shot_idx = [sparse_idx_to_dense, dense_shot_idx]
        return shot_idx


class NeighborShotSampler:
    """ This is for scene boundary detection (sbd), i.e., fine-tuning stage """

    def __init__(self, neighbor_size: int = 8):
        self.neighbor_size = neighbor_size

    def __call__(self, center_sid: int, total_num_shot: int):
        # total number of shots = 2 * neighbor_size + 1
        shot_idx = center_sid + np.arange(-self.neighbor_size,
                                          self.neighbor_size + 1)
        shot_idx = np.clip(shot_idx, 0,
                           total_num_shot)  # for out-of-boundary indices

        return shot_idx
