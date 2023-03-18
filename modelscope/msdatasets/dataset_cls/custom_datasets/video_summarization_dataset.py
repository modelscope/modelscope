# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM

import os

import h5py
import json
import numpy as np
import torch

from modelscope.msdatasets.dataset_cls.custom_datasets import \
    TorchCustomDataset


class VideoSummarizationDataset(TorchCustomDataset):

    def __init__(self, mode, opt, root_dir):
        self.mode = mode
        self.data_filename = os.path.join(root_dir, opt.dataset_file)
        self.split_filename = os.path.join(root_dir, opt.split_file)
        self.split_index = opt.split_index
        hdf = h5py.File(self.data_filename, 'r')
        self.list_frame_features, self.list_gtscores = [], []
        self.list_user_summary = []
        self.list_change_points = []
        self.list_n_frames = []
        self.list_positions = []

        with open(self.split_filename, encoding='utf-8') as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            frame_features = torch.Tensor(
                np.array(hdf[video_name + '/features']))
            gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
            user_summary = np.array(hdf[f'{video_name}/user_summary'])
            change_points = np.array(hdf[f'{video_name}/change_points'])
            n_frames = np.array(hdf[f'{video_name}/n_frames'])
            positions = np.array(hdf[f'{video_name}/picks'])

            self.list_frame_features.append(frame_features)
            self.list_gtscores.append(gtscore)
            self.list_user_summary.append(user_summary)
            self.list_change_points.append(change_points)
            self.list_n_frames.append(n_frames)
            self.list_positions.append(positions)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode + '_keys'])
        return self.len

    def __getitem__(self, index):
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        user_summary = self.list_user_summary[index]
        change_points = self.list_change_points[index]
        n_frames = self.list_n_frames[index]
        positions = self.list_positions[index]

        return dict(
            frame_features=frame_features,
            gtscore=gtscore,
            user_summary=user_summary,
            change_points=change_points,
            n_frames=n_frames,
            positions=positions)
