# Part of the implementation is borrowed and modified from PGL-SUM,
# publicly available at https://github.com/e-apostolidis/PGL-SUM, follow the
# license https://github.com/e-apostolidis/PGL-SUM/blob/master/LICENSE.md.

import os

import h5py
import json
import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.msdatasets.task_datasets.builder import TASK_DATASETS
from modelscope.msdatasets.task_datasets.torch_base_dataset import \
    TorchTaskDataset
from modelscope.utils.constant import Tasks


@TASK_DATASETS.register_module(
    Tasks.language_guided_video_summarization,
    module_name=Models.language_guided_video_summarization)
class LanguageGuidedVideoSummarizationDataset(TorchTaskDataset):

    def __init__(self, mode, opt, root_dir):
        self.mode = mode
        self.data_filename = os.path.join(root_dir, opt.dataset_file)
        self.split_filename = os.path.join(root_dir, opt.split_file)
        self.split_index = opt.split_index
        hdf = h5py.File(self.data_filename, 'r')
        self.list_image_features = []
        self.list_text_features = []
        self.list_gtscores = []
        self.list_user_summary = []
        self.list_change_points = []
        self.list_n_frames = []
        self.list_positions = []

        with open(self.split_filename) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    break

        for video_name in self.split[self.mode + '_keys']:
            clip_image_features = torch.Tensor(
                np.array(hdf[video_name + '/features_clip_image']))
            clip_txt_features = torch.Tensor(
                np.array(hdf[video_name + '/features_clip_txt'])).reshape(
                    1, -1)
            clip_txt_features = clip_txt_features.repeat(
                clip_image_features.size(0), 1)

            gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
            user_summary = np.array(hdf[f'{video_name}/user_summary'])
            change_points = np.array(hdf[f'{video_name}/change_points'])
            n_frames = np.array(hdf[f'{video_name}/n_frames'])
            positions = np.array(hdf[f'{video_name}/picks'])

            self.list_image_features.append(clip_image_features)
            self.list_text_features.append(clip_txt_features)
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
        clip_image_features = self.list_image_features[index]
        clip_txt_features = self.list_text_features[index]
        gtscore = self.list_gtscores[index]
        user_summary = self.list_user_summary[index]
        change_points = self.list_change_points[index]
        n_frames = self.list_n_frames[index]
        positions = self.list_positions[index]

        return dict(
            frame_features=clip_image_features,
            txt_features=clip_txt_features,
            gtscore=gtscore,
            user_summary=user_summary,
            change_points=change_points,
            n_frames=n_frames,
            positions=positions)
