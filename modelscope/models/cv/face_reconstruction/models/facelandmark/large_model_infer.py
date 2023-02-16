# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np
import torch

from modelscope.models.cv.skin_retouching.retinaface.predict_single import \
    Model
from ...utils import image_warp_grid1, spread_flow
from .large_base_lmks_infer import LargeBaseLmkInfer

INPUT_SIZE = 224
ENLARGE_RATIO = 1.35


def resize_on_long_side(img, long_side=800):
    src_height = img.shape[0]
    src_width = img.shape[1]

    if src_height > src_width:
        scale = long_side * 1.0 / src_height
        _img = cv2.resize(
            img, (int(src_width * scale), long_side),
            interpolation=cv2.INTER_CUBIC)

    else:
        scale = long_side * 1.0 / src_width
        _img = cv2.resize(
            img, (long_side, int(src_height * scale)),
            interpolation=cv2.INTER_CUBIC)

    return _img, scale


def draw_line(im, points, color, stroke_size=2, closed=False):
    points = points.astype(np.int32)
    for i in range(len(points) - 1):
        cv2.line(im, tuple(points[i]), tuple(points[i + 1]), color,
                 stroke_size)
    if closed:
        cv2.line(im, tuple(points[0]), tuple(points[-1]), color, stroke_size)


def enlarged_bbox(bbox, img_width, img_height, enlarge_ratio=0.2):
    '''
    :param bbox: [xmin,ymin,xmax,ymax]
    :return: bbox: [xmin,ymin,xmax,ymax]
    '''

    left = bbox[0]
    top = bbox[1]

    right = bbox[2]
    bottom = bbox[3]

    roi_width = right - left
    roi_height = bottom - top

    new_left = left - int(roi_width * enlarge_ratio)
    new_left = 0 if new_left < 0 else new_left

    new_top = top - int(roi_height * enlarge_ratio)
    new_top = 0 if new_top < 0 else new_top

    new_right = right + int(roi_width * enlarge_ratio)
    new_right = img_width if new_right > img_width else new_right

    new_bottom = bottom + int(roi_height * enlarge_ratio)
    new_bottom = img_height if new_bottom > img_height else new_bottom

    bbox = [new_left, new_top, new_right, new_bottom]

    bbox = [int(x) for x in bbox]

    return bbox


class FaceInfo:

    def __init__(self):
        self.rect = np.asarray([0, 0, 0, 0])
        self.points_array = np.zeros((106, 2))
        self.eye_left = np.zeros((22, 2))
        self.eye_right = np.zeros((22, 2))
        self.eyebrow_left = np.zeros((13, 2))
        self.eyebrow_right = np.zeros((13, 2))
        self.lips = np.zeros((64, 2))


class LargeModelInfer:

    def __init__(self, ckpt, device='cuda'):
        self.large_base_lmks_model = LargeBaseLmkInfer.model_preload(
            ckpt,
            device.lower() == 'cuda')
        self.device = device.lower()
        self.detector = Model(max_size=512, device=device)
        detector_ckpt_name = 'retinaface_resnet50_2020-07-20_old_torch.pth'
        state_dict = torch.load(
            os.path.join(os.path.dirname(ckpt), detector_ckpt_name),
            map_location='cpu')
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

    def infer(self, img_bgr):
        landmarks = []

        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.predict_jsons(rgb_image)

        boxes = []
        for anno in results:
            if anno['score'] == -1:
                break
            boxes.append({
                'x1': anno['bbox'][0],
                'y1': anno['bbox'][1],
                'x2': anno['bbox'][2],
                'y2': anno['bbox'][3]
            })

        for detect_result in boxes:
            x1 = detect_result['x1']
            y1 = detect_result['y1']
            x2 = detect_result['x2']
            y2 = detect_result['y2']

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    int(dy),
                    int(edy),
                    int(dx),
                    int(edx),
                    cv2.BORDER_CONSTANT,
                    value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))

            base_lmks = LargeBaseLmkInfer.infer_img(crop_img,
                                                    self.large_base_lmks_model,
                                                    self.device == 'cuda')

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][
                    0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][
                    1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            x1 = np.min(affine_base_lmks[:, 0])
            y1 = np.min(affine_base_lmks[:, 1])
            x2 = np.max(affine_base_lmks[:, 0])
            y2 = np.max(affine_base_lmks[:, 1])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(
                    crop_img,
                    int(dy),
                    int(edy),
                    int(dx),
                    int(edx),
                    cv2.BORDER_CONSTANT,
                    value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))

            base_lmks = LargeBaseLmkInfer.infer_img(
                crop_img, self.large_base_lmks_model,
                self.device.lower() == 'cuda')

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][
                    0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][
                    1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            landmarks.append(affine_base_lmks)

        return boxes, landmarks

    def find_face_contour(self, image):

        boxes, landmarks = self.infer(image)
        landmarks = np.array(landmarks)

        args = [[0, 33, False], [33, 38, False], [42, 47, False],
                [51, 55, False], [57, 64, False], [66, 74, True],
                [75, 83, True], [84, 96, True]]

        roi_bboxs = []

        for i in range(len(boxes)):
            roi_bbox = enlarged_bbox([
                boxes[i]['x1'], boxes[i]['y1'], boxes[i]['x2'], boxes[i]['y2']
            ], image.shape[1], image.shape[0], 0.5)
            roi_bbox = [int(x) for x in roi_bbox]
            roi_bboxs.append(roi_bbox)

        people_maps = []

        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :, :]
            maps = []
            whole_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            roi_box = roi_bboxs[i]
            roi_box_width = roi_box[2] - roi_box[0]
            roi_box_height = roi_box[3] - roi_box[1]
            short_side_length = roi_box_width if roi_box_width < roi_box_height else roi_box_height

            line_width = short_side_length // 10

            if line_width == 0:
                line_width = 1

            kernel_size = line_width * 2
            gaussian_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            for t, arg in enumerate(args):
                mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                draw_line(mask, landmark[arg[0]:arg[1]], (255, 255, 255),
                          line_width, arg[2])
                mask = cv2.GaussianBlur(mask,
                                        (gaussian_kernel, gaussian_kernel), 0)
                if t >= 1:
                    draw_line(whole_mask, landmark[arg[0]:arg[1]],
                              (255, 255, 255), line_width * 2, arg[2])
                maps.append(mask)
            whole_mask = cv2.GaussianBlur(whole_mask,
                                          (gaussian_kernel, gaussian_kernel),
                                          0)
            maps.append(whole_mask)
            people_maps.append(maps)

        return people_maps[0], boxes

    def face2contour(self, image, stack_mode='column'):
        '''

        :param facer:
        :param image:
        :param stack_mode:
        :return: final_maps: [map0, map1,....]
                 roi_bboxs: [bbox0, bbox1, ...]
        '''

        boxes, landmarks = self.infer(image)
        landmarks = np.array(landmarks)

        args = [[0, 33, False], [33, 38, False], [42, 47, False],
                [51, 55, False], [57, 64, False], [66, 74, True],
                [75, 83, True], [84, 96, True]]

        roi_bboxs = []

        for i in range(len(boxes)):
            roi_bbox = enlarged_bbox([
                boxes[i]['x1'], boxes[i]['y1'], boxes[i]['x2'], boxes[i]['y2']
            ], image.shape[1], image.shape[0], 0.5)
            roi_bbox = [int(x) for x in roi_bbox]
            roi_bboxs.append(roi_bbox)

        people_maps = []

        for i in range(landmarks.shape[0]):
            landmark = landmarks[i, :, :]
            maps = []
            whole_mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)

            roi_box = roi_bboxs[i]
            roi_box_width = roi_box[2] - roi_box[0]
            roi_box_height = roi_box[3] - roi_box[1]
            short_side_length = roi_box_width if roi_box_width < roi_box_height else roi_box_height

            line_width = short_side_length // 50

            if line_width == 0:
                line_width = 1

            kernel_size = line_width * 4
            gaussian_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

            for arg in args:
                mask = np.zeros((image.shape[0], image.shape[1]), np.uint8)
                draw_line(mask, landmark[arg[0]:arg[1]], (255, 255, 255),
                          line_width, arg[2])
                mask = cv2.GaussianBlur(mask,
                                        (gaussian_kernel, gaussian_kernel), 0)
                draw_line(whole_mask, landmark[arg[0]:arg[1]], (255, 255, 255),
                          line_width, arg[2])
                maps.append(mask)
            whole_mask = cv2.GaussianBlur(whole_mask,
                                          (gaussian_kernel, gaussian_kernel),
                                          0)
            maps.append(whole_mask)
            people_maps.append(maps)

        if stack_mode == 'depth':
            final_maps = []
            for i, maps in enumerate(people_maps):
                final_map = np.dstack(maps)
                final_map = final_map[roi_bboxs[i][1]:roi_bboxs[i][3],
                                      roi_bboxs[i][0]:roi_bboxs[i][2], :]
                final_maps.append(final_map)
            return final_maps, roi_bboxs

        elif stack_mode == 'column':
            final_maps = []
            for i, maps in enumerate(people_maps):
                joint_maps = [
                    x[roi_bboxs[i][1]:roi_bboxs[i][3],
                      roi_bboxs[i][0]:roi_bboxs[i][2]] for x in maps
                ]
                final_map = np.column_stack(joint_maps)
                final_maps.append(final_map)
            return final_maps, roi_bboxs

    def fat_face(self, img, degree=0.1):

        _img, scale = resize_on_long_side(img, 800)

        contour_maps, boxes = self.find_face_contour(_img)

        contour_map = contour_maps[0]

        boxes = boxes[0]

        Flow = np.zeros(
            shape=(contour_map.shape[0], contour_map.shape[1], 2),
            dtype=np.float32)

        box_center = [(boxes['x1'] + boxes['x2']) / 2,
                      (boxes['y1'] + boxes['y2']) / 2]

        box_length = max(
            abs(boxes['y1'] - boxes['y2']), abs(boxes['x1'] - boxes['x2']))

        value_1 = 2 * (Flow.shape[0] - box_center[1] - 1)
        value_2 = 2 * (Flow.shape[1] - box_center[0] - 1)
        value_list = [
            box_length * 2, 2 * (box_center[0] - 1), 2 * (box_center[1] - 1),
            value_1, value_2
        ]
        flow_box_length = min(value_list)
        flow_box_length = int(flow_box_length)

        sf = spread_flow(100, flow_box_length * degree)
        sf = cv2.resize(sf, (flow_box_length, flow_box_length))

        Flow[int(box_center[1]
                 - flow_box_length / 2):int(box_center[1]
                                            + flow_box_length / 2),
             int(box_center[0]
                 - flow_box_length / 2):int(box_center[0]
                                            + flow_box_length / 2)] = sf

        Flow = Flow * np.dstack((contour_map, contour_map)) / 255.0

        inter_face_maps = contour_maps[-1]

        Flow = Flow * (1.0 - np.dstack(
            (inter_face_maps, inter_face_maps)) / 255.0)

        Flow = cv2.resize(Flow, (img.shape[1], img.shape[0]))

        Flow = Flow / scale

        pred, top_bound, bottom_bound, left_bound, right_bound = image_warp_grid1(
            Flow[..., 0], Flow[..., 1], img, 1.0, [0, 0, 0, 0])

        return pred
