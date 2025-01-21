# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import random

import decord
import numpy as np
import torch
from detectron2.data.transforms import (ExtentTransform, RandomBrightness,
                                        RandomFlip, ResizeShortestEdge)
from detectron2.structures import Boxes, Instances
from scipy.interpolate import interp1d


def inp_boxes(boxes: dict, start, end):
    idxs = sorted([int(i) for i in boxes.keys()])
    bbox = [boxes[str(i)] for i in idxs]
    new_bboxes = []
    for i in range(4):
        f = interp1d(idxs, [b[i] for b in bbox])
        new_b = f(list(range(start, end + 1)))
        new_bboxes.append(new_b)
    new_bboxes = np.stack(new_bboxes, axis=1)
    return new_bboxes


def assign_label(start, end, data_dict):
    """
    根据视频起始位置，以及标注的label,给这小段视频安排bbox检测标签
    方法，取交集，交集占到样本的一半或者标签的一半，即将该label赋给样本
    :param start: 起始帧号（含）
    :param end: 结束帧号（含）
    :param labels: 标注的label, 字符串形式
    :return:[[行为，x1,y1,x2,y2],]
    """
    if 'actions' not in data_dict:
        return []
    scale = data_dict['scale']
    gt_labels = []
    for action in data_dict['actions']:
        low = max(int(action['start']), start)
        high = min(int(action['end']), end)
        inter = 0 if low > high else high - low
        if inter > (end - start) * 0.7 or inter > (action['end']
                                                   - action['start']) * 0.7:
            boxes = inp_boxes(action['boxes'], low, high)
            box = boxes.mean(axis=0) / scale
            label = [action['label']] + box.tolist()
            gt_labels.append(label)
    return gt_labels


class VideoDetMapper:

    def __init__(self,
                 classes_id_map,
                 used_seconds=2,
                 input_frames=4,
                 is_train=True,
                 tile=False):
        self.classes_id = classes_id_map
        self.is_train = is_train
        self.used_seconds = used_seconds
        self.input_frames = input_frames
        self.tile = tile
        self.trans = [RandomBrightness(0.5, 1.5)]
        self.tfm_gens = [
            ResizeShortestEdge((480, 512, 544, 576, 608, 640, 672, 704, 736,
                                768) if is_train else 512,
                               1280 if is_train else 896, 'choice')
        ]
        if is_train:
            self.tfm_gens.append(RandomFlip())

    def __call__(self, data_dict):
        data_dict = copy.deepcopy(data_dict)
        try:
            data_dict = self._call(data_dict)
        except Exception as e:
            print(data_dict['path:FILE'], e)
            data_dict = None
        return data_dict

    def _call(self, data_dict):
        video_name = data_dict['path:FILE']
        if data_dict['actions'] is not None:
            data_dict['actions'] = eval(data_dict['actions'])
        else:
            data_dict['actions'] = []

        v = decord.VideoReader(video_name, ctx=decord.cpu(0))
        num_frames = len(v)
        used_frames = max(int((1 + random.random()) * v.get_avg_fps()), 1)
        if self.is_train:
            start_idx = random.randint(0, max(0, num_frames - used_frames))
        else:
            start_idx = max(0, num_frames - used_frames) // 2
        idxs = np.linspace(start_idx, min(start_idx + used_frames, num_frames) - 1, self.input_frames) \
            .round().astype('int32').tolist()
        imgs = v.get_batch(idxs).asnumpy()
        del v
        labels = assign_label(idxs[0], idxs[-1] + 1, data_dict)
        bboxes = np.array([label[-4:] for label in labels])

        if self.is_train:
            if self.tile:
                imgs, labels, bboxes = self.random_tile(
                    video_name, imgs, labels, bboxes, pos_choices=[1, 1, 2, 4])
            else:
                imgs, labels, bboxes = self.random_tile(
                    video_name, imgs, labels, bboxes, pos_choices=[1])

            for g in self.trans:
                tfm = g.get_transform(imgs)
                imgs = tfm.apply_image(imgs)
            imgs, bboxes = self.random_extent(imgs, bboxes)

        for trans in self.tfm_gens:
            tfm = trans.get_transform(imgs[0])
            imgs = np.stack([tfm.apply_image(img) for img in imgs])
            bboxes = tfm.apply_box(bboxes)

        _, h, w, c = imgs.shape
        data_dict['height'] = h
        data_dict['width'] = w
        gt_boxes = Boxes(torch.from_numpy(bboxes))  # XYXY_ABS
        gt_classes = [self.classes_id[label[0]]
                      for label in labels]  # N is background
        instances = Instances((data_dict['height'], data_dict['width']))
        instances.set('gt_boxes', gt_boxes)
        instances.set('gt_classes',
                      torch.as_tensor(gt_classes, dtype=torch.int64))
        data_dict['instances'] = instances
        data_dict['frames'] = torch.as_tensor(
            np.ascontiguousarray(imgs.transpose([3, 0, 1, 2])))
        return data_dict

    def random_tile(self, name, imgs, labels, bboxes,
                    pos_choices=(1, 1, 2, 4)):
        _, h, w, c = imgs.shape
        bboxes = bboxes.tolist()
        if len(labels) == 0:  # 负样本 1/2, 1, 2, 4
            ratio = random.choice([0, 1, 2, 4])
            if ratio == 0:  # 随机取部分区域
                h0, w0 = random.randint(0, h // 2), random.randint(0, w // 2)
                imgs = imgs[:, h0:h0 + h // 2, w0:w0 + h // 2]
            elif ratio == 2:
                imgs = np.tile(imgs,
                               (1, 1, 2,
                                1)) if h > w else np.tile(imgs, (1, 2, 1, 1))
            elif ratio == 4:
                imgs = np.tile(imgs, (1, 2, 2, 1))
        else:  # 正样本 1, 2, 4
            ratio = random.choice(pos_choices)
            if ratio == 2:
                labels = labels * 2
                if h >= w:  # 左右拼接
                    imgs = np.tile(imgs, (1, 1, 2, 1))
                    bbox2 = [[x1 + w, y1, x2 + w, y2]
                             for x1, y1, x2, y2 in bboxes]
                else:  # 上下拼接
                    imgs = np.tile(imgs, (1, 2, 1, 1))
                    bbox2 = [[x1, y1 + h, x2, y2 + h]
                             for x1, y1, x2, y2 in bboxes]
                bboxes = bboxes + bbox2
            elif ratio == 4:
                labels = labels * 4
                imgs = np.tile(imgs, (1, 2, 2, 1))
                bbox2 = [[x1 + w, y1, x2 + w, y2] for x1, y1, x2, y2 in bboxes] + \
                        [[x1, y1 + h, x2, y2 + h] for x1, y1, x2, y2 in bboxes] + \
                        [[x1 + w, y1 + h, x2 + w, y2 + h] for x1, y1, x2, y2 in bboxes]
                bboxes = bboxes + bbox2
        bboxes = np.array(bboxes)
        return imgs.copy(), labels, bboxes

    def random_extent(self, imgs, bboxes):
        t, h, w, c = imgs.shape
        r_h, r_w = int(h * 0.1), int(w * 0.1)
        x0, y0 = random.randint(-r_w, r_w), random.randint(-r_h, r_h)
        x1, y1 = random.randint(w - r_w,
                                w + r_w), random.randint(h - r_h, h + r_h)
        tfm = ExtentTransform((x0, y0, x1, y1), output_size=(y1 - y0, x1 - x0))
        imgs = np.stack([tfm.apply_image(img) for img in imgs])
        bboxes = tfm.apply_box(bboxes)
        return imgs, bboxes
