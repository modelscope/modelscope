# -*- coding: utf-8 -*

import os
import numpy as np
from PIL import Image, ImageFile
import cv2 as cv
import torch
from torch.utils.data.dataset import Dataset
from .utils import merge_bboxes
from torch.utils.data.dataloader import default_collate
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --------------------------------------------------- #
#   获得所有分类
# --------------------------------------------------- #
def get_classes(classes_path):
    """
    :param classes_path: 分类.txt的路径
    :return: (所有分类组成的列表，所有锚框组成的narray(锚框.shape=[3, 3, 2]))
    """
    assert os.path.isfile(classes_path), "路径 {} 不存在!".format(classes_path)
    with open(classes_path) as f:
        lines = f.readlines()
    class_names = []
    for line in lines:
        line = line.strip()   # 去掉每一行的前后空格符
        if line:
            class_names.append(line)

    return class_names


# --------------------------------------------------- #
#   获得先验框
# --------------------------------------------------- #
def get_anchors(anchors_path):
    """
    :param anchors_path: 锚框.txt的路径
    :return: (所有分类组成的列表，所有锚框组成的narray(锚框.shape=[3, 3, 2]))
    """
    assert os.path.isfile(anchors_path), "路径 {} 不存在!".format(anchors_path)
    with open(anchors_path) as f:
        lines = f.readlines()
    anchors = []
    for line in lines:
        line = line.strip()   # 去掉每一行的前后空格符
        if line:
            anchors.append(line.split(','))
    anchors = np.array(anchors, dtype="float").reshape([-1, 3, 2])[::-1, :, :]     # 从大到小排列

    return anchors


# ------------------------------------ #
#   将batch中数据转换为ndarray,DataLoader中collate_fn使用
# ------------------------------------ #
def yolo_dataset_collate(batch):
    """
    :param batch: batch中每个元素形如(data, label)
    :return:
    """
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = default_collate(images)   # 内部使用stack将含tensor的列表拼接成tensor
    return images, bboxes


class YoloDataset(Dataset):
    def __init__(self, train_lines, input_size, mosaic=True, transfer_gray=False, in_channels=3):
        super(YoloDataset, self).__init__()
        self.train_lines = train_lines
        self.input_size = input_size
        self.mosaic = mosaic
        self.transfer_gray = transfer_gray
        self.in_channels = in_channels
        self.flag = True

    def __len__(self):
        return len(self.train_lines)

    def __getitem__(self, index):
        # 返回的数据都是numpy格式
        if self.mosaic:
            if self.flag and (index + 4) < len(self.train_lines):
                img, y = self.get_random_data_with_Mosaic(self.train_lines[index:index + 4])
            else:
                img, y = self.get_random_data(self.train_lines[index])
            self.flag = bool(1 - self.flag)
        else:
            img, y = self.get_random_data(self.train_lines[index])

        if len(y) != 0:
            # 从坐标转换成0~1的百分比
            boxes = np.array(y[:, :4], dtype=np.float32)
            boxes[:, 0] = boxes[:, 0] / self.input_size[1]
            boxes[:, 1] = boxes[:, 1] / self.input_size[0]
            boxes[:, 2] = boxes[:, 2] / self.input_size[1]
            boxes[:, 3] = boxes[:, 3] / self.input_size[0]

            boxes = np.maximum(np.minimum(boxes, 1), 0)
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

            boxes[:, 0] = boxes[:, 0] + boxes[:, 2] / 2
            boxes[:, 1] = boxes[:, 1] + boxes[:, 3] / 2
            y = np.concatenate([boxes, y[:, -1:]], axis=-1)

        if self.transfer_gray and self.in_channels == 1:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if self.in_channels == 3:
                img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        img = np.array(img, dtype=np.float32)

        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))    # 将h*w*c转换为c*h*w格式
        tmp_targets = np.array(y, dtype=np.float32)
        tmp_inp = torch.from_numpy(tmp_inp).type(torch.FloatTensor)   # 转换为torch.float32类型
        tmp_targets = torch.from_numpy(tmp_targets).type(torch.FloatTensor)

        return tmp_inp, tmp_targets

    def rand(self, a=0.0, b=1.0):    # 生成a~b之间的随机数
        # np.random.seed(10101)
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, jitter=0.3, hue=0.1, sat=1.5, val=1.5):
        """实时数据增强的随机预处理"""
        line = annotation_line.split()
        image = Image.open(line[0]).convert("RGB")   # 不转换则为RGBA通道图像
        iw, ih = image.size
        h, w = self.input_size[:2]
        box = np.array([list(map(int, box.split(','))) for box in line[1:]])

        # 调整图片大小
        new_ar = w / h * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
        scale = self.rand(0.5, 1.5)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 放置图片
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h),
                              (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 是否翻转图片
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # 色域变换
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv.cvtColor(np.array(image, np.float32) / 255, cv.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv.cvtColor(x, cv.COLOR_HSV2RGB) * 255

        # 调整目标框坐标
        box_data = np.zeros((len(box), 5))
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]  # 保留有效框
            box_data = np.zeros((len(box), 5))
            box_data[:len(box)] = box
        if len(box) == 0:
            return image_data, []

        if (box_data[:, :4] > 0).any():
            return image_data, box_data
        else:
            return image_data, []

    def get_random_data_with_Mosaic(self, annotation_line, hue=0.1, sat=1.5, val=1.5):
        h, w = self.input_size[:2]
        min_offset_x = 0.3
        min_offset_y = 0.3
        scale_low = 1 - min(min_offset_x, min_offset_y)
        scale_high = scale_low + 0.2

        image_datas = []
        box_datas = []
        index = 0

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        for line in annotation_line:
            # 每一行进行分割
            line_content = line.split()
            # 打开图片
            image = Image.open(line_content[0]).convert("RGB")   # 不转换则为RGBA通道图像
            # 图片的大小
            iw, ih = image.size
            # 保存框的位置
            box = np.array([list(map(int, box.split(','))) for box in line_content[1:]])

            # 是否翻转图片
            flip = self.rand() < 0.5    # 0.5的概率翻转
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # 对输入进来的图片进行缩放
            new_ar = w / h
            scale = self.rand(scale_low, scale_high)    # 随机缩放系数
            if new_ar < 1:   # h大，主要是保持宽高比不变
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:   # w大或者w=h
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # 进行色域变换
            hue = self.rand(-hue, hue)
            sat = self.rand(1, sat) if self.rand() < 0.5 else 1 / self.rand(1, sat)
            val = self.rand(1, val) if self.rand() < 0.5 else 1 / self.rand(1, val)
            x = cv.cvtColor(np.array(image, np.float32) / 255, cv.COLOR_RGB2HSV)
            x[..., 0] += hue * 360
            x[..., 0][x[..., 0] > 1] -= 1
            x[..., 0][x[..., 0] < 0] += 1
            x[..., 1] *= sat
            x[..., 2] *= val
            x[x[:, :, 0] > 360, 0] = 360
            x[:, :, 1:][x[:, :, 1:] > 1] = 1
            x[x < 0] = 0
            image = cv.cvtColor(x, cv.COLOR_HSV2RGB)      # numpy array, 0 to 1

            image = Image.fromarray((image * 255).astype(np.uint8))
            # 将图片进行放置，分别对应四张分割图片的位置
            dx = place_x[index]
            dy = place_y[index]
            new_image = Image.new('RGB', (w, h),
                                  (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # 对box进行重新处理
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # 将图片分割，放在一起
        cutx = np.random.randint(int(w * min_offset_x), int(w * (1 - min_offset_x)))
        cuty = np.random.randint(int(h * min_offset_y), int(h * (1 - min_offset_y)))

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        # 对框进行进一步的处理
        new_boxes = np.array(merge_bboxes(box_datas, cutx, cuty))

        if len(new_boxes) == 0:
            return new_image, []
        if (new_boxes[:, :4] > 0).any():
            return new_image, new_boxes
        else:
            return new_image, []
