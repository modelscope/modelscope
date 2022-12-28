# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp

import numpy as np
from mmcls.datasets.base_dataset import BaseDataset


def get_trained_checkpoints_name(work_path):
    import os
    file_list = os.listdir(work_path)
    last = 0
    model_name = None
    # find the best model
    if model_name is None:
        for f_name in file_list:
            if 'best_' in f_name and f_name.endswith('.pth'):
                best_epoch = f_name.replace('.pth', '').split('_')[-1]
                if best_epoch.isdigit():
                    last = int(best_epoch)
                    model_name = f_name
                    return model_name
    # or find the latest model
    if model_name is None:
        for f_name in file_list:
            if 'epoch_' in f_name and f_name.endswith('.pth'):
                epoch_num = f_name.replace('epoch_', '').replace('.pth', '')
                if not epoch_num.isdigit():
                    continue
                ind = int(epoch_num)
                if ind > last:
                    last = ind
                    model_name = f_name
    return model_name


def preprocess_transform(cfgs):
    if cfgs is None:
        return None
    for i, cfg in enumerate(cfgs):
        if cfg.type == 'Resize':
            if isinstance(cfg.size, list):
                cfgs[i].size = tuple(cfg.size)
    return cfgs


def get_ms_dataset_root(ms_dataset):
    if ms_dataset is None or len(ms_dataset) < 1:
        return None
    try:
        data_root = ms_dataset[0]['image:FILE'].split('extracted')[0]
        path_post = ms_dataset[0]['image:FILE'].split('extracted')[1].split(
            '/')
        extracted_data_root = osp.join(data_root, 'extracted', path_post[1],
                                       path_post[2])
        return extracted_data_root
    except Exception as e:
        raise ValueError(f'Dataset Error: {e}')
    return None


def get_classes(classes=None):
    import mmcv
    if isinstance(classes, str):
        # take it as a file path
        class_names = mmcv.list_from_file(classes)
    elif isinstance(classes, (tuple, list)):
        class_names = classes
    else:
        raise ValueError(f'Unsupported type {type(classes)} of classes.')

    return class_names


class MmDataset(BaseDataset):

    def __init__(self, ms_dataset, pipeline, classes=None, test_mode=False, data_prefix=''):
        self.ms_dataset = ms_dataset
        if len(self.ms_dataset) < 1:
            raise ValueError('Dataset Error: dataset is empty')
        super(MmDataset, self).__init__(
            data_prefix=data_prefix,
            pipeline=pipeline,
            classes=classes,
            test_mode=test_mode)

    def load_annotations(self):
        if self.CLASSES is None:
            raise ValueError(
                f'Dataset Error: Not found classesname.txt: {self.CLASSES}')

        data_infos = []
        for data_info in self.ms_dataset:
            filename = data_info['image:FILE']
            gt_label = data_info['category']
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            data_infos.append(info)

        return data_infos
