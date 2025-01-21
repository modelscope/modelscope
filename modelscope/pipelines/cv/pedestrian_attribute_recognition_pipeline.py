# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from typing import Any, Dict, List, Union

import cv2
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.pedestrian_attribute_recognition.model import \
    PedestrainAttribute
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.pedestrian_attribute_recognition,
    module_name=Pipelines.pedestrian_attribute_recognition)
class PedestrainAttributeRecognitionPipeline(Pipeline):
    """ Pedestrian attribute recognition Pipeline.

    Example:

    ```python
    >>> from modelscope.pipelines import pipeline
    >>> from modelscope.utils.constant import Tasks

    >>> model_id = 'damo/cv_resnet50_pedestrian-attribute-recognition_image'
    >>> handle = pipeline(Tasks.pedestrian_attribute_recognition, model=model_id)
    >>> output = handle('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/keypoints_detect/000000442836.jpg')
    ```
    """

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        """
        use `model` to create a image depth estimation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        self.attribute_model = PedestrainAttribute(num_classes=39)
        state = torch.load(
            osp.join(model, ModelFile.TORCH_MODEL_FILE),
            map_location=self.device)
        self.attribute_model.load_state_dict(state)
        self.attribute_model = self.attribute_model.to(self.device)
        self.attribute_model.eval()

        self.input_size = [192, 384]
        self.box_enlarge_ratio = 0

        self.human_detect_model_id = 'damo/cv_tinynas_human-detection_damoyolo'
        self.human_detector = pipeline(
            Tasks.domain_specific_object_detection,
            model=self.human_detect_model_id)

    def get_labels(self, outputs, thres=0.5):
        gender = outputs[0][0:1]
        age = outputs[0][1:4]
        orient = outputs[0][4:7]
        hat = outputs[0][7:8]
        glass = outputs[0][8:9]
        hand_bag = outputs[0][9:10]
        shoulder_bag = outputs[0][10:11]
        back_pack = outputs[0][11:12]
        upper_wear = outputs[0][12:14]
        lower_wear = outputs[0][14:17]
        upper_color = outputs[0][17:28]
        lower_color = outputs[0][28:39]

        lb_gender = 0 if gender > thres else 1
        lb_age = np.argmax(age)
        lb_orient = np.argmax(orient)
        lb_hat = 0 if hat > thres else 1
        lb_glass = 0 if glass > thres else 1
        lb_hand_bag = 0 if hand_bag > thres else 1
        lb_shoulder_bag = 0 if shoulder_bag > thres else 1
        lb_back_pack = 0 if back_pack > thres else 1
        lb_upper_wear = np.argmax(upper_wear)
        lb_lower_wear = np.argmax(lower_wear)
        lb_upper_color = np.argmax(upper_color)
        lb_lower_color = np.argmax(lower_color)
        labels = [
            lb_gender, lb_age, lb_orient, lb_hat, lb_glass, lb_hand_bag,
            lb_shoulder_bag, lb_back_pack, lb_upper_wear, lb_lower_wear,
            lb_upper_color, lb_lower_color
        ]

        return labels

    def labels_transform(self, labels):
        notes_en = [
            ['Female', 'Male'],
            ['AgeOver60', 'Age18-60', 'AgeLess18'],
            ['Front', 'Side', 'Back'],
            ['Yes', 'No'],
            ['Yes', 'No'],
            ['Yes', 'No'],
            ['Yes', 'No'],
            ['Yes', 'No'],
            ['ShortSleeve', 'LongSleeve'],
            ['Trousers', 'Shorts', 'Skirt&Dress'],
            [
                'black', 'grey', 'blue', 'green', 'white', 'purple', 'red',
                'brown', 'yellow', 'pink', 'orange'
            ],
            [
                'black', 'grey', 'blue', 'green', 'white', 'purple', 'red',
                'brown', 'yellow', 'pink', 'orange'
            ],
        ]
        notes_cn = [
            ['女', '男'],
            ['大于60岁', '18-60岁之间', '小于18岁'],
            ['正向', '侧向', '背面'],
            ['戴帽子', '不戴帽子'],
            ['戴眼镜', '不戴眼镜'],
            ['有手提包', '无手提包'],
            ['有肩挎包', '无肩挎包'],
            ['有背包', '无背包'],
            ['短袖', '长袖'],
            ['长裤', '短裤', '裙子'],
            ['黑', '灰', '蓝', '绿', '白', '紫', '红', '棕', '黄', '粉', '橙'],
            ['黑', '灰', '蓝', '绿', '白', '紫', '红', '棕', '黄', '粉', '橙'],
        ]

        notes_labels_en = []
        notes_labels_cn = []

        for idx, lb in enumerate(labels):
            notes_labels_en.append(notes_en[idx][lb])
            notes_labels_cn.append(notes_cn[idx][lb])

        return notes_labels_en, notes_labels_cn

    def get_results(self, inputs: Dict[Tensor, Dict[str, np.ndarray]],
                    **kwargs):
        output_labels = []
        output_boxes = []

        for i in range(len(inputs[0])):
            outputs = self.get_labels((inputs[0][i]).detach().cpu().numpy())
            label_en, label_cn = self.labels_transform(outputs)
            box = np.array(inputs[1][i]['human_box'][0:4]).reshape(2, 2)

            output_labels.append(label_en)
            output_boxes.append(box.tolist())

        return output_boxes, output_labels

    def image_transform(self, input: Input) -> Dict[Tensor, Any]:
        if isinstance(input, str):
            image = cv2.imread(input, -1)[:, :, 0:3]
        elif isinstance(input, np.ndarray):
            if len(input.shape) == 2:
                image = cv2.cvtColor(input, cv2.COLOR_GRAY2BGR)
            else:
                image = input
            image = image[:, :, 0:3]
        elif isinstance(input, torch.Tensor):
            image = input.cpu().numpy()[:, :, 0:3]

        w_new = self.input_size[0]
        h_new = self.input_size[1]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = cv2.resize(image, (w_new, h_new), cv2.INTER_LINEAR)

        img_resize = np.float32(img_resize) / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_resize = (img_resize - mean) / std

        input_data = np.zeros([1, 3, h_new, w_new], dtype=np.float32)

        img_resize = img_resize.transpose((2, 0, 1))
        input_data[0, :] = img_resize
        return torch.from_numpy(input_data)

    def crop_image(self, image, box):
        height, width, _ = image.shape
        w, h = box[1] - box[0]
        box[0, :] -= (w * self.box_enlarge_ratio, h * self.box_enlarge_ratio)
        box[1, :] += (w * self.box_enlarge_ratio, h * self.box_enlarge_ratio)

        box[0, 0] = min(max(box[0, 0], 0.0), width)
        box[0, 1] = min(max(box[0, 1], 0.0), height)
        box[1, 0] = min(max(box[1, 0], 0.0), width)
        box[1, 1] = min(max(box[1, 1], 0.0), height)

        cropped_image = image[int(box[0][1]):int(box[1][1]),
                              int(box[0][0]):int(box[1][0])]
        return cropped_image

    def process_image(self, input: Dict[Tensor, Tensor]) -> Dict[Tensor, Any]:
        bboxes = input[0]
        image = input[1]

        lst_human_images = []
        lst_meta = []
        for i in range(len(bboxes)):
            box = np.array(bboxes[i][0:4]).reshape(2, 2)
            box[1] += box[0]
            human_image = self.crop_image(image.clone(), box)
            meta = {}
            human_image = self.image_transform(human_image)
            lst_human_images.append(human_image)
            meta['human_box'] = box
            lst_meta.append(meta)

        return [lst_human_images, lst_meta]

    def preprocess(self, input: Input) -> Dict[Tensor, Union[str, np.ndarray]]:
        output = self.human_detector(input)

        image = LoadImage.convert_to_ndarray(input)
        image = image[:, :, [2, 1, 0]]  # rgb2bgr

        return {'image': image, 'output': output}

    def forward(self, input: Tensor) -> Dict[Tensor, Dict[str, np.ndarray]]:
        input_image = input['image']
        output = input['output']

        bboxes = []
        scores = np.array(output[OutputKeys.SCORES].cpu(), dtype=np.float32)
        boxes = np.array(output[OutputKeys.BOXES].cpu(), dtype=np.float32)

        for id, box in enumerate(boxes):
            box_tmp = [
                box[0], box[1], box[2] - box[0], box[3] - box[1], scores[id], 0
            ]
            bboxes.append(box_tmp)
        if len(bboxes) == 0:
            logger.error('cannot detect human in the image')
            return [None, None]
        human_images, metas = self.process_image([bboxes, input_image])

        outputs = []
        for image in human_images:
            output = self.attribute_model.forward(image.to(self.device))
            output = torch.sigmoid(output)
            outputs.append(output)

        return [outputs, metas]

    def postprocess(self, input: Dict[Tensor, Dict[str, np.ndarray]],
                    **kwargs) -> str:
        if input[0] is None or input[1] is None:
            return {OutputKeys.BOXES: [], OutputKeys.LABELS: []}

        boxes, labels = self.get_results(input)
        result_boxes = []
        for box in boxes:
            result_boxes.append([box[0][0], box[0][1], box[1][0], box[1][1]])
        return {OutputKeys.BOXES: result_boxes, OutputKeys.LABELS: labels}
