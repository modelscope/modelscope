# Copyright (c) Alibaba, Inc. and its affiliates.

from typing import Any, Dict
import torch
import numpy as np
import cv2
import os
import colorsys
from PIL import ImageFile
from PIL import Image, ImageFont, ImageDraw
from modelscope.metainfo import Pipelines
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from modelscope.pipelines.cv.tbs_detection_utils.utils import non_max_suppression, DecodeBox, yolo_correct_boxes

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = get_logger()

__all__ = ['TBSDetectionPipeline']

@PIPELINES.register_module(
    Tasks.image_object_detection, module_name=Pipelines.tbs_detection)
class TBSDetectionPipeline(Pipeline):

    _defaults = {
        "model_image_size": (416, 416, 3),
        "confidence": 0.1,
        "iou": 0.9,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"


    def __init__(self, model: str, **kwargs):
        """
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        self.__dict__.update(self._defaults)
        self.class_names, self.num_classes = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()


    # ---------------------------------------------------#
    #   获得所有的分类
    # ---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.join(self.model, 'model_data/voc_classes.txt')
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)

    # ---------------------------------------------------#
    #   获得所有的先验框
    # ---------------------------------------------------#
    def _get_anchors(self):
        anchors_path = os.path.join(self.model, 'model_data/yolo_anchors.txt')
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            lines = f.readlines()
        anchors = [line.strip().split(',') for line in lines]

        return np.array(anchors, dtype="float").reshape([-1, 3, 2])[::-1, :, :]

    def generate(self):

        self.yolo_decodes = []
        for i in range(len(self.anchors)):
            self.yolo_decodes.append(
                DecodeBox(self.anchors[i], len(self.class_names), self.model_image_size[:2][::-1]))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.) for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

    # --------------------------------------------------- #
    #   后处理
    # --------------------------------------------------- #
    def post_process(self, outputs, img_path):
        new_boxes = []
        output_list = []
        top_confs = 0
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                               conf_thres=self.confidence,
                                               nms_thres=self.iou)

        for j, batch_detection in enumerate(batch_detections):
            if batch_detection is None:
                continue
            try:
                batch_detection = batch_detection.cpu().numpy()
            except:
                return

            image = Image.open(img_path)
            image_shape = np.array(np.shape(image)[0:2])
            top_index = batch_detection[:, 4] * batch_detection[:, 5] > self.confidence
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
                                       np.array(self.model_image_size[:2]), image_shape)

            # print(boxes)
            font_path = os.path.join(self.model, 'model_data/simhei.ttf')
            font = ImageFont.truetype(font=font_path, size=int(3e-2 * image.size[0] + 0.5))  # 字体大小
            thickness = (image.size[1] + image.size[0]) // self.model_image_size[0]  # 框大小
            # new_boxes = []
            for i, c in enumerate(top_label):
                top, left, bottom, right = boxes[i]
                top = max(0, round(top, 2))

                left = max(0, round(left, 2))
                bottom = min(image.size[1], round(bottom, 2))
                right = min(image.size[0], round(right, 2))
                new_boxes.append([top, left, bottom, right])

                # 画框框
                predicted_class = self.class_names[c]
                score = (top_conf * top_class)[i]
                label = '{} {:.2f}'.format(predicted_class, score)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for t in range(thickness):
                    draw.rectangle((left + t, top + t, right - t, bottom - t),
                                   outline=self.colors[self.class_names.index(predicted_class)])
                draw.rectangle((tuple(text_origin), tuple(text_origin + label_size)),
                               fill=self.colors[self.class_names.index(predicted_class)])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

                save_path = "./draw_box"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                img_name = os.path.basename(img_path)  # 图片名字
                save_img_path = os.path.join(save_path, img_name)
                image.save(save_img_path)

        return new_boxes, top_confs

    def preprocess(self, input: Input) -> Dict[str, Any]:

        img = LoadImage.convert_to_ndarray(input)
        img = img.astype(np.float)
        result = {'img': img, 'img_path': input}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        img = input['img'].astype(np.uint8)
        img = cv2.resize(img, (416, 416))
        img = img.astype(np.float32)
        tmp_inp = np.transpose(img / 255.0, (2, 0, 1))
        tmp_inp = torch.from_numpy(tmp_inp).type(torch.FloatTensor)
        img = torch.unsqueeze(tmp_inp, dim=0)
        model_path = os.path.join(self.model, 'pytorch_yolov4.pt')
        model = torch.load(model_path)
        outputs = model(img.cuda())
        result = {'data': outputs ,'img_path': input['img_path']}
        return result

    def postprocess(self, input: Dict[str, Any], *args, **kwargs) -> Dict[str, Any]:

        bboxes, scores = self.post_process(input['data'], input['img_path'])

        if bboxes is None:
            outputs = {
                OutputKeys.SCORES: [],
                OutputKeys.BOXES: []
            }
            return outputs
        outputs = {
            OutputKeys.SCORES: scores.tolist(),
            OutputKeys.LABELS: [],
            OutputKeys.BOXES: bboxes
        }
        return outputs

