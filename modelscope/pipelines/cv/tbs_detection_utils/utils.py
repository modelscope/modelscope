from __future__ import division
import colorsys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from PIL import Image
from torchvision.ops.boxes import batched_nms, nms

plt.switch_backend('Agg')


class DecodeBox(nn.Module):

    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.img_size = img_size

        self.num_anchors = len(anchors)
        self.bbox_attrs = 5 + num_classes

    def forward(self, input):
        # input为bs,3*(1+4+num_classes),13,13
        # 一共多少张图片
        batch_size = input.size(0)
        # 13，13
        input_height = input.size(2)
        input_width = input.size(3)

        # 计算步长
        # 每一个特征点对应原来的图片上多少个像素点
        # 如果特征层为13x13的话，一个特征点就对应原来的图片上的32个像素点
        # 416/13 = 32
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        # 把先验框的尺寸调整成特征层大小的形式
        # 计算出先验框在特征层上对应的宽高
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h)
                          for anchor_width, anchor_height in self.anchors]

        # bs,3*(5+num_classes),13,13 -> bs,3,13,13,(5+num_classes)
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs,
                                input_height,
                                input_width).permute(0, 1, 3, 4,
                                                     2).contiguous()

        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        # 获得置信度，是否有物体
        conf = torch.sigmoid(prediction[..., 4])
        # 种类置信度
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Cls pred.
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角 batch_size,3,13,13
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(
            input_width, 1).repeat(batch_size * self.num_anchors, 1,
                                   1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(
            input_height, 1).t().repeat(batch_size * self.num_anchors, 1,
                                        1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(
            1, 1, input_height * input_width).view(h.shape)
        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # 用于将输出调整为相对于416x416的大小
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,
                            conf.view(batch_size, -1, 1),
                            pred_cls.view(batch_size, -1, self.num_classes)),
                           -1)

        return output.data


# ------------------------------------------------- #
#   输入图片的尺寸为正方形，而数据集中的图片一般为长方形，粗暴的resize会使得图片失真，采用letterbox可以较好的解决这个问题
#   该方法可以保持图片的长宽比例，剩下的部分采用灰色填充
# ------------------------------------------------- #
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image


# ------------------------------------------------- #
#   对模型输出的box信息(x, y, w, h)进行校正,输出基于原图坐标系的box信息(x_min, y_min, x_max, y_max)
# ------------------------------------------------- #
def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    """
    :param top: 模型输出的box中心坐标信息,范围0~1
    :param left: 模型输出的box中心坐标信息,范围0~1
    :param bottom: 模型输出的box长宽信息,范围0~1
    :param right: 模型输出的box长宽信息,范围0~1
    :param input_shape: 模型的图像尺寸, 长宽均是32倍数
    :param image_shape: 原图尺寸
    :return: 基于原图坐标系的box信息(实际坐标值,非比值)
    """
    new_shape = image_shape * np.min(input_shape / image_shape)
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape
    box_yx = np.concatenate(
        ((top + bottom) / 2, (left + right) / 2), axis=-1) / input_shape
    box_hw = np.concatenate(
        (bottom - top, right - left), axis=-1) / input_shape
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = [
        box_mins[:, 0:1], box_mins[:, 1:2], box_maxes[:, 0:1], box_maxes[:,
                                                                         1:2]
    ]
    boxes = np.concatenate(boxes, axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)

    return boxes


# ------------------------------------------------- #
#   计算IOU
# ------------------------------------------------- #
def bbox_iou(box1, box2, x1y1x2y2=True):
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:,
                                                                  2], box1[:,
                                                                           3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:,
                                                                  2], box2[:,
                                                                           3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(
        inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0)
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


# ------------------------------------------------- #
#   非极大值抑制
# ------------------------------------------------- #
def non_max_suppression(prediction,
                        num_classes,
                        conf_thres=0.5,
                        nms_thres=0.4):
    # 求左上角和右下角
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # 获得种类及其置信度
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # 利用置信度进行第一轮筛选
        score = image_pred[:, 4] * class_conf[:, 0]
        conf_mask = (score >= conf_thres).squeeze()

        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        # 获得的内容为(x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        # 获得种类
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            # 获得某一类初步筛选后全部的预测结果
            detections_class = detections[detections[:, -1] == c]

            # ------------------------------------------ #
            #   使用官方自带的非极大抑制会速度更快一些！
            # ------------------------------------------ #
            keep = nms(detections_class[:, :4],
                       detections_class[:, 4] * detections_class[:, 5],
                       nms_thres)
            max_detections = detections_class[keep]

            output[image_i] = max_detections if output[
                image_i] is None else torch.cat(
                    [output[image_i], max_detections])

    return output


# ------------------------------------------------- #
#   合并boxes
# ------------------------------------------------- #
def merge_bboxes(bboxes, cutx, cuty):
    merge_bbox = []
    for i in range(len(bboxes)):
        for box in bboxes[i]:
            tmp_box = []
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if i == 0:
                if y1 > cuty or x1 > cutx:
                    continue
                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue
                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 1:
                if y2 < cuty or x1 > cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x2 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 2:
                if y2 < cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y1 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            if i == 3:
                if y1 > cuty or x2 < cutx:
                    continue

                if y2 >= cuty and y1 <= cuty:
                    y2 = cuty
                    if y2 - y1 < 5:
                        continue

                if x2 >= cutx and x1 <= cutx:
                    x1 = cutx
                    if x2 - x1 < 5:
                        continue

            tmp_box.append(x1)
            tmp_box.append(y1)
            tmp_box.append(x2)
            tmp_box.append(y2)
            tmp_box.append(box[-1])
            merge_bbox.append(tmp_box)
    return merge_bbox


# ---------------------------------------------------#
#   获得所有的先验框
# ---------------------------------------------------#
def _get_anchors(self):
    anchors_path = os.path.join(self.model, 'model_data/yolo_anchors.txt')
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        lines = f.readlines()
    anchors = [line.strip().split(',') for line in lines]
    return np.array(anchors, dtype='float').reshape([-1, 3, 2])[::-1, :, :]


def generate(self):
    self.yolo_decodes = []
    for i in range(len(self.anchors)):
        self.yolo_decodes.append(
            DecodeBox(self.anchors[i], len(self.class_names),
                      self.model_image_size[:2][::-1]))

    # 画框设置不同的颜色
    hsv_tuples = [(x / len(self.class_names), 1., 1.)
                  for x in range(len(self.class_names))]
    self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    self.colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            self.colors))


# --------------------------------------------------- #
#   后处理
# --------------------------------------------------- #
def post_process(self, outputs, img_path):
    new_boxes = []
    output_list = []
    top_confs = torch.empty(0)
    for i in range(3):
        output_list.append(self.yolo_decodes[i](outputs[i]))
    output = torch.cat(output_list, 1)
    batch_detections = non_max_suppression(
        output,
        len(self.class_names),
        conf_thres=self.confidence,
        nms_thres=self.iou)

    for j, batch_detection in enumerate(batch_detections):
        if batch_detection is None:
            continue
        try:
            batch_detection = batch_detection.cpu().numpy()
        except Exception:
            return

        image = Image.open(img_path)
        image_shape = np.array(np.shape(image)[0:2])
        top_index = batch_detection[:,
                                    4] * batch_detection[:,
                                                         5] > self.confidence
        top_conf = batch_detection[top_index, 4]
        top_class = batch_detection[top_index, 5]
        top_confs = top_conf * top_class
        top_label = np.array(batch_detection[top_index, -1], np.int32)
        top_bboxes = np.array(batch_detection[top_index, :4])
        top_xmin = np.expand_dims(top_bboxes[:, 0], -1)
        top_ymin = np.expand_dims(top_bboxes[:, 1], -1)
        top_xmax = np.expand_dims(top_bboxes[:, 2], -1)
        top_ymax = np.expand_dims(top_bboxes[:, 3], -1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                   np.array(self.model_image_size[:2]),
                                   image_shape)

        for i, c in enumerate(top_label):
            top, left, bottom, right = boxes[i]
            top = max(0, round(top, 2))

            left = max(0, round(left, 2))
            bottom = min(image.size[1], round(bottom, 2))
            right = min(image.size[0], round(right, 2))
            new_boxes.append([top, left, bottom, right])

    return new_boxes, top_confs
