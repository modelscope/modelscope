# ------------------------------------------------------------------------------
# Part of implementation is adopted from DBNet,
# made publicly available under the Apache License 2.0 at https://github.com/MhLiao/DB.
# ------------------------------------------------------------------------------
import bisect
import functools
import glob
import logging
import math
import os

import cv2
import numpy as np
import torch.utils.data as data

from .processes import (AugmentDetectionData, MakeBorderMap, MakeICDARData,
                        MakeSegDetectionData, NormalizeImage, RandomCropData)


class ImageDataset(data.Dataset):
    r'''Dataset reading from images.
    '''

    def __init__(self, cfg, data_dir=None, data_list=None, **kwargs):
        self.data_dir = data_dir
        self.data_list = data_list
        if 'train' in self.data_list[0]:
            self.is_training = True
        else:
            self.is_training = False
        self.image_paths = []
        self.gt_paths = []
        self.get_all_samples()
        self.processes = None
        if self.is_training and hasattr(cfg.train, 'transform'):
            self.processes = cfg.train.transform
        elif not self.is_training and hasattr(cfg.test, 'transform'):
            self.processes = cfg.test.transform

    def get_all_samples(self):
        for i in range(len(self.data_dir)):
            with open(self.data_list[i], 'r') as fid:
                image_list = fid.readlines()
                fid.close()
            if self.is_training:
                image_path = [
                    self.data_dir[i] + '/train_images/' + timg.strip()
                    for timg in image_list
                ]
                gt_path = [
                    self.data_dir[i] + '/train_gts/'
                    + timg.strip().split('.')[0] + '.txt'
                    for timg in image_list
                ]
            else:
                image_path = [
                    self.data_dir[i] + '/test_images/' + timg.strip()
                    for timg in image_list
                ]
                gt_path = [
                    self.data_dir[i] + '/test_gts/'
                    + timg.strip().split('.')[0] + '.txt'
                    for timg in image_list
                ]
            self.image_paths += image_path
            self.gt_paths += gt_path
        self.num_samples = len(self.image_paths)
        self.targets = self.load_ann()
        if self.is_training:
            assert len(self.image_paths) == len(self.targets)

    def load_ann(self):
        res = []
        for gt in self.gt_paths:
            lines = []
            reader = open(gt, 'r')
            for line in reader.readlines():
                item = {}
                line = line.strip().split(',')
                label = line[-1]
                poly = np.array(list(map(float, line[:8]))).reshape(
                    (-1, 2)).tolist()
                item['poly'] = poly
                item['text'] = label
                lines.append(item)
            reader.close()
            res.append(lines)
        return res

    def __getitem__(self, index, retry=0):
        if index >= self.num_samples:
            index = index % self.num_samples
        data = {}
        image_path = self.image_paths[index]
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        if self.is_training:
            data['filename'] = image_path
            data['data_id'] = image_path
        else:
            data['filename'] = image_path.split('/')[-1]
            data['data_id'] = image_path.split('/')[-1]
        data['image'] = img
        target = self.targets[index]
        data['lines'] = target

        # processes in line-up way, defined in configuration.json
        if self.processes is not None:
            # normal detection augment
            if hasattr(self.processes, 'detection_augment'):
                data_process0 = AugmentDetectionData(
                    self.processes.detection_augment)
                data = data_process0(data)

            # random crop augment
            if hasattr(self.processes, 'random_crop'):
                data_process1 = RandomCropData(self.processes.random_crop)
                data = data_process1(data)

            # data build in ICDAR format
            if hasattr(self.processes, 'MakeICDARData'):
                data_process2 = MakeICDARData()
                data = data_process2(data)

            # Making binary mask from detection data with ICDAR format
            if hasattr(self.processes, 'MakeSegDetectionData'):
                data_process3 = MakeSegDetectionData()
                data = data_process3(data)

            # Making the border map from detection data with ICDAR format
            if hasattr(self.processes, 'MakeBorderMap'):
                data_process4 = MakeBorderMap()
                data = data_process4(data)

            # Image Normalization
            if hasattr(self.processes, 'NormalizeImage'):
                data_process5 = NormalizeImage()
                data = data_process5(data)

        if self.is_training:
            # remove redundant data key for training
            for key in [
                    'polygons', 'filename', 'shape', 'ignore_tags',
                    'is_training'
            ]:
                del data[key]

        return data

    def __len__(self):
        return len(self.image_paths)
