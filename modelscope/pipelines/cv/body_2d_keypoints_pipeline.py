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
from modelscope.models.cv.body_2d_keypoints.hrnet_v2 import \
    PoseHighResolutionNetV2
from modelscope.models.cv.body_2d_keypoints.w48 import cfg_128x128_15
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Model, Pipeline, Tensor
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.body_2d_keypoints, module_name=Pipelines.body_2d_keypoints)
class Body2DKeypointsPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)
        device = torch.device(
            f'cuda:{0}' if torch.cuda.is_available() else 'cpu')
        self.keypoint_model = KeypointsDetection(model, device)

        self.human_detect_model_id = 'damo/cv_resnet18_human-detection'
        self.human_detector = pipeline(
            Tasks.human_detection, model=self.human_detect_model_id)

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
        human_images, metas = self.keypoint_model.preprocess(
            [bboxes, input_image])
        outputs = self.keypoint_model.forward(human_images)
        return [outputs, metas]

    def postprocess(self, input: Dict[Tensor, Dict[str, np.ndarray]],
                    **kwargs) -> str:
        if input[0] is None or input[1] is None:
            return {
                OutputKeys.BOXES: [],
                OutputKeys.KEYPOINTS: [],
                OutputKeys.SCORES: []
            }

        poses, scores, boxes = self.keypoint_model.postprocess(input)
        result_boxes = []
        for box in boxes:
            result_boxes.append([box[0][0], box[0][1], box[1][0], box[1][1]])
        return {
            OutputKeys.BOXES: result_boxes,
            OutputKeys.KEYPOINTS: poses,
            OutputKeys.SCORES: scores
        }


class KeypointsDetection():

    def __init__(self, model: str, device: str, **kwargs):
        self.model = model
        self.device = device
        cfg = cfg_128x128_15
        self.key_points_model = PoseHighResolutionNetV2(cfg)
        pretrained_state_dict = torch.load(
            osp.join(self.model, ModelFile.TORCH_MODEL_FILE),
            map_location=device)
        self.key_points_model.load_state_dict(
            pretrained_state_dict, strict=False)
        self.key_points_model = self.key_points_model.to(device)
        self.key_points_model.eval()

        self.input_size = cfg['MODEL']['IMAGE_SIZE']
        self.lst_parent_ids = cfg['DATASET']['PARENT_IDS']
        self.lst_left_ids = cfg['DATASET']['LEFT_IDS']
        self.lst_right_ids = cfg['DATASET']['RIGHT_IDS']
        self.box_enlarge_ratio = 0.05

    def train(self):
        return self.key_points_model.train()

    def eval(self):
        return self.key_points_model.eval()

    def forward(self, input: Tensor) -> Tensor:
        with torch.no_grad():
            return self.key_points_model.forward(input.to(self.device))

    def get_pts(self, heatmaps):
        [pts_num, height, width] = heatmaps.shape
        pts = []
        scores = []
        for i in range(pts_num):
            heatmap = heatmaps[i, :, :]
            pt = np.where(heatmap == np.max(heatmap))
            scores.append(np.max(heatmap))
            x = pt[1][0]
            y = pt[0][0]

            [h, w] = heatmap.shape
            if x >= 1 and x <= w - 2 and y >= 1 and y <= h - 2:
                x_diff = heatmap[y, x + 1] - heatmap[y, x - 1]
                y_diff = heatmap[y + 1, x] - heatmap[y - 1, x]
                x_sign = 0
                y_sign = 0
                if x_diff < 0:
                    x_sign = -1
                if x_diff > 0:
                    x_sign = 1
                if y_diff < 0:
                    y_sign = -1
                if y_diff > 0:
                    y_sign = 1
                x = x + x_sign * 0.25
                y = y + y_sign * 0.25

            pts.append([x, y])
        return pts, scores

    def pts_transform(self, meta, pts, lt_x, lt_y):
        pts_new = []
        s = meta['s']
        o = meta['o']
        size = len(pts)
        for i in range(size):
            ratio = 4
            x = (int(pts[i][0] * ratio) - o[0]) / s[0]
            y = (int(pts[i][1] * ratio) - o[1]) / s[1]

            pt = [x, y]
            pts_new.append(pt)

        return pts_new

    def postprocess(self, inputs: Dict[Tensor, Dict[str, np.ndarray]],
                    **kwargs):
        output_poses = []
        output_scores = []
        output_boxes = []
        for i in range(inputs[0].shape[0]):
            outputs, scores = self.get_pts(
                (inputs[0][i]).detach().cpu().numpy())
            outputs = self.pts_transform(inputs[1][i], outputs, 0, 0)
            box = np.array(inputs[1][i]['human_box'][0:4]).reshape(2, 2)
            outputs = np.array(outputs) + box[0]
            output_poses.append(outputs.tolist())
            output_scores.append(scores)
            output_boxes.append(box.tolist())
        return output_poses, output_scores, output_boxes

    def image_crop_resize(self, input, margin=[0, 0]):
        pad_img = np.zeros((self.input_size[1], self.input_size[0], 3),
                           dtype=np.uint8)

        h, w, ch = input.shape

        h_new = self.input_size[1] - margin[1] * 2
        w_new = self.input_size[0] - margin[0] * 2
        s0 = float(h_new) / h
        s1 = float(w_new) / w
        s = min(s0, s1)
        w_new = int(s * w)
        h_new = int(s * h)

        img_new = cv2.resize(input, (w_new, h_new), cv2.INTER_LINEAR)

        cx = self.input_size[0] // 2
        cy = self.input_size[1] // 2

        pad_img[cy - h_new // 2:cy - h_new // 2 + h_new,
                cx - w_new // 2:cx - w_new // 2 + w_new, :] = img_new

        return pad_img, np.array([cx, cy]), np.array([s, s]), np.array(
            [cx - w_new // 2, cy - h_new // 2])

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

        w, h, _ = image.shape
        w_new = self.input_size[0]
        h_new = self.input_size[1]

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize, c, s, o = self.image_crop_resize(image)

        img_resize = np.float32(img_resize) / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_resize = (img_resize - mean) / std

        input_data = np.zeros([1, 3, h_new, w_new], dtype=np.float32)

        img_resize = img_resize.transpose((2, 0, 1))
        input_data[0, :] = img_resize
        meta = {'c': c, 's': s, 'o': o}
        return [torch.from_numpy(input_data), meta]

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

    def preprocess(self, input: Dict[Tensor, Tensor]) -> Dict[Tensor, Any]:
        bboxes = input[0]
        image = input[1]

        lst_human_images = []
        lst_meta = []
        for i in range(len(bboxes)):
            box = np.array(bboxes[i][0:4]).reshape(2, 2)
            box[1] += box[0]
            human_image = self.crop_image(image.clone(), box)
            human_image, meta = self.image_transform(human_image)
            lst_human_images.append(human_image)
            meta['human_box'] = box
            lst_meta.append(meta)

        return [torch.cat(lst_human_images, dim=0), lst_meta]
