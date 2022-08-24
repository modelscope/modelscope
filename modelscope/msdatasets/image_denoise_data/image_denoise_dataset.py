import os
from typing import Callable, List, Optional, Tuple, Union

import cv2
import numpy as np
from torch.utils import data

from .data_utils import img2tensor, padding, paired_paths_from_folder
from .transforms import augment, paired_random_crop


def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


class PairedImageDataset(data.Dataset):
    """Paired image dataset for image restoration.
    """

    def __init__(self, opt, root, is_train):
        super(PairedImageDataset, self).__init__()
        self.opt = opt
        self.is_train = is_train
        self.gt_folder, self.lq_folder = os.path.join(
            root, opt.dataroot_gt), os.path.join(root, opt.dataroot_lq)

        if opt.filename_tmpl is not None:
            self.filename_tmpl = opt.filename_tmpl
        else:
            self.filename_tmpl = '{}'
        self.paths = paired_paths_from_folder([self.lq_folder, self.gt_folder],
                                              ['lq', 'gt'], self.filename_tmpl)

    def __getitem__(self, index):
        scale = self.opt.scale

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]['gt_path']
        img_gt = default_loader(gt_path)
        lq_path = self.paths[index]['lq_path']
        img_lq = default_loader(lq_path)

        # augmentation for training
        # if self.is_train:
        gt_size = self.opt.gt_size
        # padding
        img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        # random crop
        img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale)

        # flip, rotation
        img_gt, img_lq = augment([img_gt, img_lq], self.opt.use_flip,
                                 self.opt.use_rot)

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt, img_lq = img2tensor([img_gt, img_lq],
                                    bgr2rgb=True,
                                    float32=True)

        return {
            'input': img_lq,
            'target': img_gt,
            'input_path': lq_path,
            'target_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def to_torch_dataset(
        self,
        columns: Union[str, List[str]] = None,
        preprocessors: Union[Callable, List[Callable]] = None,
        **format_kwargs,
    ):
        return self
