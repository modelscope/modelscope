# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, 'wb')
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(
            image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception(
            'Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1],
                                image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


S_H, S_W = 0, 0


class MVSDataset(Dataset):

    def __init__(self, root_dir, list_file, mode, n_views, **kwargs):
        super(MVSDataset, self).__init__()

        self.root_dir = root_dir
        self.list_file = list_file
        self.mode = mode
        self.n_views = n_views

        assert self.mode in ['train', 'val', 'test']

        self.total_depths = 192
        self.interval_scale = 1.06

        self.data_scale = kwargs.get('data_scale', 'mid')  # mid / raw
        self.robust_train = kwargs.get('robust_train', False)  # True / False
        self.color_augment = transforms.ColorJitter(
            brightness=0.5, contrast=0.5)

        if self.mode == 'test':
            self.max_wh = kwargs.get('max_wh', (1600, 1200))
            self.max_w, self.max_h = self.max_wh

        self.fix_res = kwargs.get(
            'fix_res', False)  # whether to fix the resolution of input image.
        self.fix_wh = False

        # self.metas = self.build_metas()
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.list_file
        # logger.info("MVSDataset scans:", scans)

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = '{}/pair.txt'.format(scan)
            # read the pair file
            with open(os.path.join(self.root_dir, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [
                        int(x) for x in f.readline().rstrip().split()[1::2]
                    ]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.n_views:
                            src_views += [src_views[0]] * (
                                self.n_views - len(src_views))
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(
            ' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(
            ' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.total_depths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        if self.mode == 'train' and self.robust_train:
            img = self.color_augment(img)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        return np_img

    def crop_img(self, img):
        raw_h, raw_w = img.shape[:2]
        start_h = (raw_h - 1024) // 2
        start_w = (raw_w - 1280) // 2
        return img[start_h:start_h + 1024,
                   start_w:start_w + 1280, :]  # (1024, 1280)

    def prepare_img(self, hr_img):
        h, w = hr_img.shape
        if self.data_scale == 'mid':
            hr_img_ds = cv2.resize(
                hr_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST)
            h, w = hr_img_ds.shape
            target_h, target_w = 512, 640
            start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
            hr_img_crop = hr_img_ds[start_h:start_h + target_h,
                                    start_w:start_w + target_w]
        elif self.data_scale == 'raw':
            hr_img_crop = hr_img[h // 2 - 1024 // 2:h // 2 + 1024 // 2,
                                 w // 2 - 1280 // 2:w // 2
                                 + 1280 // 2]  # (1024, 1280)
        return hr_img_crop

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        if h > max_h or w > max_w:
            scale = 1.0 * max_h / h
            if scale * w > max_w:
                scale = 1.0 * max_w / w
            new_w, new_h = scale * w // base * base, scale * h // base * base
        else:
            new_w, new_h = 1.0 * w // base * base, 1.0 * h // base * base

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = self.prepare_img(np_img)

        h, w = np_img.shape
        np_img_ms = {
            'stage1':
            cv2.resize(
                np_img, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            'stage2':
            cv2.resize(
                np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            'stage3':
            cv2.resize(
                np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            'stage4':
            np_img,
        }
        return np_img_ms

    def read_depth_hr(self, filename, scale):
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32) * scale
        depth_lr = self.prepare_img(depth_hr)

        h, w = depth_lr.shape
        depth_lr_ms = {
            'stage1':
            cv2.resize(
                depth_lr, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            'stage2':
            cv2.resize(
                depth_lr, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            'stage3':
            cv2.resize(
                depth_lr, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            'stage4':
            depth_lr,
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        global S_H, S_W
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.n_views - 1]

        scale_ratio = 1

        imgs = []
        depth_values = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(
                self.root_dir, '{}/images_post/{:0>8}.jpg'.format(scan, vid))
            if not os.path.exists(img_filename):
                img_filename = os.path.join(
                    self.root_dir, '{}/images/{:0>8}.jpg'.format(scan, vid))

            proj_mat_filename = os.path.join(
                self.root_dir, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(
                proj_mat_filename,
                interval_scale=self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w,
                                                   self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                S_H, S_W = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    S_H, S_W = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != S_H) or (c_w != S_W):
                scale_h = 1.0 * S_H / c_h
                scale_w = 1.0 * S_W / c_w
                img = cv2.resize(img, (S_W, S_H))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            #################
            imgs.append(img.transpose(2, 0, 1))

            # reference view
            if i == 0:
                # @Note depth values
                diff = 0.5 if self.mode in ['test', 'val'] else 0
                depth_max = depth_interval * (self.total_depths
                                              - diff) + depth_min
                depth_values = np.array(
                    [depth_min * scale_ratio, depth_max * scale_ratio],
                    dtype=np.float32)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

        proj_matrices = np.stack(proj_matrices)
        intrinsics = np.stack(intrinsics)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2.0
        stage1_ins = intrinsics.copy()
        stage1_ins[:2, :] = intrinsics[:2, :] / 2.0
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_ins = intrinsics.copy()
        stage3_ins[:2, :] = intrinsics[:2, :] * 2.0
        stage4_pjmats = proj_matrices.copy()
        stage4_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
        stage4_ins = intrinsics.copy()
        stage4_ins[:2, :] = intrinsics[:2, :] * 4.0
        proj_matrices = {
            'stage1': stage1_pjmats,
            'stage2': proj_matrices,
            'stage3': stage3_pjmats,
            'stage4': stage4_pjmats
        }
        intrinsics_matrices = {
            'stage1': stage1_ins,
            'stage2': intrinsics,
            'stage3': stage3_ins,
            'stage4': stage4_ins
        }

        sample = {
            'imgs': imgs,
            'proj_matrices': proj_matrices,
            'intrinsics_matrices': intrinsics_matrices,
            'depth_values': depth_values,
            'filename': scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + '{}'
        }
        return sample
