import os
import sys
import cv2
import math
import copy
import hashlib
import imageio
import numpy as np
import pandas as pd
from scipy import interpolate
from PIL import Image, ImageEnhance, ImageFile

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .encoder import get_encoder


class AlignmentDataset(Dataset):

    def __init__(self, tsv_flie, image_dir="", transform=None,
                 width=256, height=256, channels=3,
                 means=(127.5, 127.5, 127.5), scale=1 / 127.5,
                 classes_num=None, crop_op=True, aug_prob=0.0, edge_info=None, flip_mapping=None, is_train=True,
                 encoder_type='default',
                 ):
        super(AlignmentDataset, self).__init__()
        self.use_AAM = True
        self.encoder_type = encoder_type
        self.encoder = get_encoder(height, width, encoder_type=encoder_type)
        self.items = pd.read_csv(tsv_flie, sep="\t")
        self.image_dir = image_dir
        self.landmark_num = classes_num[0]
        self.transform = transform

        self.image_width = width
        self.image_height = height
        self.channels = channels
        assert self.image_width == self.image_height

        self.means = means
        self.scale = scale

        self.aug_prob = aug_prob
        self.edge_info = edge_info
        self.is_train = is_train
        std_lmk_5pts = np.array([
            196.0, 226.0,
            316.0, 226.0,
            256.0, 286.0,
            220.0, 360.4,
            292.0, 360.4], np.float32) / 256.0 - 1.0
        std_lmk_5pts = np.reshape(std_lmk_5pts, (5, 2))  # [-1 1]
        target_face_scale = 1.0 if crop_op else 1.25

        self.augmentation = Augmentation(
            is_train=self.is_train,
            aug_prob=self.aug_prob,
            image_size=self.image_width,
            crop_op=crop_op,
            std_lmk_5pts=std_lmk_5pts,
            target_face_scale=target_face_scale,
            flip_rate=0.5,
            flip_mapping=flip_mapping,
            random_shift_sigma=0.05,
            random_rot_sigma=math.pi / 180 * 18,
            random_scale_sigma=0.1,
            random_gray_rate=0.2,
            random_occ_rate=0.4,
            random_blur_rate=0.3,
            random_gamma_rate=0.2,
            random_nose_fusion_rate=0.2)

    def _circle(self, img, pt, sigma=1.0, label_type='Gaussian'):
        # Check that any part of the gaussian is in-bounds
        tmp_size = sigma * 3
        ul = [int(pt[0] - tmp_size), int(pt[1] - tmp_size)]
        br = [int(pt[0] + tmp_size + 1), int(pt[1] + tmp_size + 1)]
        if (ul[0] > img.shape[1] - 1 or ul[1] > img.shape[0] - 1 or
                br[0] - 1 < 0 or br[1] - 1 < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        if label_type == 'Gaussian':
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        else:
            g = sigma / (((x - x0) ** 2 + (y - y0) ** 2 + sigma ** 2) ** 1.5)

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = 255 * g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    def _polylines(self, img, lmks, is_closed, color=255, thickness=1, draw_mode=cv2.LINE_AA,
                   interpolate_mode=cv2.INTER_AREA, scale=4):
        h, w = img.shape
        img_scale = cv2.resize(img, (w * scale, h * scale), interpolation=interpolate_mode)
        lmks_scale = (lmks * scale + 0.5).astype(np.int32)
        cv2.polylines(img_scale, [lmks_scale], is_closed, color, thickness * scale, draw_mode)
        img = cv2.resize(img_scale, (w, h), interpolation=interpolate_mode)
        return img

    def _generate_edgemap(self, points, scale=0.25, thickness=1):
        h, w = self.image_height, self.image_width
        edgemaps = []
        for is_closed, indices in self.edge_info:
            edgemap = np.zeros([h, w], dtype=np.float32)
            # align_corners: False.
            part = copy.deepcopy(points[np.array(indices)])

            part = self._fit_curve(part, is_closed)
            part[:, 0] = np.clip(part[:, 0], 0, w - 1)
            part[:, 1] = np.clip(part[:, 1], 0, h - 1)
            edgemap = self._polylines(edgemap, part, is_closed, 255, thickness)

            edgemaps.append(edgemap)
        edgemaps = np.stack(edgemaps, axis=0) / 255.0
        edgemaps = torch.from_numpy(edgemaps).float().unsqueeze(0)
        edgemaps = F.interpolate(edgemaps, size=(int(w * scale), int(h * scale)), mode='bilinear',
                                 align_corners=False).squeeze()
        return edgemaps

    def _fit_curve(self, lmks, is_closed=False, density=5):
        try:
            x = lmks[:, 0].copy()
            y = lmks[:, 1].copy()
            if is_closed:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
            tck, u = interpolate.splprep([x, y], s=0, per=is_closed, k=3)
            # bins = (x.shape[0] - 1) * density + 1
            # lmk_x, lmk_y = interpolate.splev(np.linspace(0, 1, bins), f)
            intervals = np.array([])
            for i in range(len(u) - 1):
                intervals = np.concatenate((intervals, np.linspace(u[i], u[i + 1], density, endpoint=False)))
            if not is_closed:
                intervals = np.concatenate((intervals, [u[-1]]))
            lmk_x, lmk_y = interpolate.splev(intervals, tck, der=0)
            # der_x, der_y = interpolate.splev(intervals, tck, der=1)
            curve_lmks = np.stack([lmk_x, lmk_y], axis=-1)
            # curve_ders = np.stack([der_x, der_y], axis=-1)
            # origin_indices = np.arange(0, curve_lmks.shape[0], density)

            return curve_lmks
        except:
            return lmks

    def _image_id(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, image_path)
        return hashlib.md5(open(image_path, "rb").read()).hexdigest()

    def _load_image(self, image_path):
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_dir, image_path)

        try:
            # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)#HWC, BGR, [0-255]
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # HWC, BGR, [0-255]
            assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
        except:
            try:
                img = imageio.imread(image_path)  # HWC, RGB, [0-255]
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
            except:
                try:
                    gifImg = imageio.mimread(image_path)  # BHWC, RGB, [0-255]
                    img = gifImg[0]  # HWC, RGB, [0-255]
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # HWC, BGR, [0-255]
                    assert img is not None and len(img.shape) == 3 and img.shape[2] == 3
                except:
                    img = None
        return img

    def _compose_rotate_and_scale(self, angle, scale, shift_xy, from_center, to_center):
        cosv = math.cos(angle)
        sinv = math.sin(angle)

        fx, fy = from_center
        tx, ty = to_center

        acos = scale * cosv
        asin = scale * sinv

        a0 = acos
        a1 = -asin
        a2 = tx - acos * fx + asin * fy + shift_xy[0]

        b0 = asin
        b1 = acos
        b2 = ty - asin * fx - acos * fy + shift_xy[1]

        rot_scale_m = np.array([
            [a0, a1, a2],
            [b0, b1, b2],
            [0.0, 0.0, 1.0]
        ], np.float32)
        return rot_scale_m

    def _transformPoints2D(self, points, matrix):
        """
        points (nx2), matrix (3x3) -> points (nx2)
        """
        dtype = points.dtype

        # nx3
        points = np.concatenate([points, np.ones_like(points[:, [0]])], axis=1)
        points = points @ np.transpose(matrix)  # nx3
        points = points[:, :2] / points[:, [2, 2]]
        return points.astype(dtype)

    def _transformPerspective(self, image, matrix, target_shape):
        """
        image, matrix3x3 -> transformed_image
        """
        return cv2.warpPerspective(
            image, matrix,
            dsize=(target_shape[1], target_shape[0]),
            flags=cv2.INTER_LINEAR, borderValue=0)

    def _norm_points(self, points, h, w, align_corners=False):
        if align_corners:
            # [0, SIZE-1] -> [-1, +1]
            des_points = points / torch.tensor([w - 1, h - 1]).to(points).view(1, 2) * 2 - 1
        else:
            # [-0.5, SIZE-0.5] -> [-1, +1]
            des_points = (points * 2 + 1) / torch.tensor([w, h]).to(points).view(1, 2) - 1
        des_points = torch.clamp(des_points, -1, 1)
        return des_points

    def _denorm_points(self, points, h, w, align_corners=False):
        if align_corners:
            # [-1, +1] -> [0, SIZE-1]
            des_points = (points + 1) / 2 * torch.tensor([w - 1, h - 1]).to(points).view(1, 1, 2)
        else:
            # [-1, +1] -> [-0.5, SIZE-0.5]
            des_points = ((points + 1) * torch.tensor([w, h]).to(points).view(1, 1, 2) - 1) / 2
        return des_points

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        sample = dict()

        image_path = self.items.iloc[index, 0]
        landmarks_5pts = self.items.iloc[index, 1]
        landmarks_5pts = np.array(list(map(float, landmarks_5pts.split(","))), dtype=np.float32).reshape(5, 2)
        landmarks_target = self.items.iloc[index, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            self.landmark_num, 2)
        scale = float(self.items.iloc[index, 3])
        center_w, center_h = float(self.items.iloc[index, 4]), float(self.items.iloc[index, 5])
        if len(self.items.iloc[index]) > 6:
            tags = np.array(list(map(lambda x: int(float(x)), self.items.iloc[index, 6].split(","))))
        else:
            tags = np.array([])

        # image & keypoints alignment
        image_path = image_path.replace('\\', '/')
        # wflw testset
        image_path = image_path.replace(
            '//msr-facestore/Workspace/MSRA_EP_Allergan/users/yanghuan/training_data/wflw/rawImages/', '')
        # trainset
        image_path = image_path.replace('./rawImages/', '')
        image_path = os.path.join(self.image_dir, image_path)

        # image path
        sample["image_path"] = image_path

        img = self._load_image(image_path)  # HWC, BGR, [0, 255]
        assert img is not None

        # augmentation
        # landmarks_target = [-0.5, edge-0.5]
        img, landmarks_target, matrix = \
            self.augmentation.process(img, landmarks_target, landmarks_5pts, scale, center_w, center_h)

        landmarks = self._norm_points(torch.from_numpy(landmarks_target), self.image_height, self.image_width)

        sample["label"] = [landmarks, ]

        if self.use_AAM:
            pointmap = self.encoder.generate_heatmap(landmarks_target)
            edgemap = self._generate_edgemap(landmarks_target)
            sample["label"] += [pointmap, edgemap]

        sample['matrix'] = matrix

        # image normalization
        img = img.transpose(2, 0, 1).astype(np.float32)  # CHW, BGR, [0, 255]
        img[0, :, :] = (img[0, :, :] - self.means[0]) * self.scale
        img[1, :, :] = (img[1, :, :] - self.means[1]) * self.scale
        img[2, :, :] = (img[2, :, :] - self.means[2]) * self.scale
        sample["data"] = torch.from_numpy(img)  # CHW, BGR, [-1, 1]

        sample["tags"] = tags

        return sample
