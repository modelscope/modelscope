# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import json
import numpy as np
import tensorflow as tf

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()


class HeadSegmentor():

    def __init__(self, model_root):
        """The HeadSegmentor is implemented based on https://arxiv.org/abs/2004.04955
        Args:
            model_root: the root directory of the model files
        """
        self.sess = self.load_sess(
            os.path.join(model_root, 'head_segmentation',
                         'Matting_headparser_6_18.pb'))
        self.sess_detect = self.load_sess(
            os.path.join(model_root, 'head_segmentation', 'face_detect.pb'))
        self.sess_face = self.load_sess(
            os.path.join(model_root, 'head_segmentation', 'segment_face.pb'))

    def load_sess(self, model_path):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with tf.gfile.FastGFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
            sess.run(tf.global_variables_initializer())
        return sess

    def process(self, image):
        """ image: bgr
        """

        h, w, c = image.shape
        faceRects = self.detect_face(image)
        face_num = len(faceRects)
        all_head_alpha = []
        all_face_mask = []
        for i in range(face_num):
            y1 = faceRects[i][0]
            y2 = faceRects[i][1]
            x1 = faceRects[i][2]
            x2 = faceRects[i][3]
            pad_y1, pad_y2, pad_x1, pad_x2 = self.pad_box(
                y1, y2, x1, x2, 0.15, 0.15, 0.15, 0.15, h, w)
            temp_img = image.copy()
            roi_img = temp_img[pad_y1:pad_y2, pad_x1:pad_x2]
            output_alpha = self.sess_face.run(
                self.sess_face.graph.get_tensor_by_name('output_alpha_face:0'),
                feed_dict={'input_image_face:0': roi_img[:, :, ::-1]})
            face_mask = np.zeros((h, w, 3))
            face_mask[pad_y1:pad_y2, pad_x1:pad_x2] = output_alpha
            all_face_mask.append(face_mask)
            cv2.imwrite(str(i) + 'face.jpg', face_mask)
            cv2.imwrite(str(i) + 'face_roi.jpg', roi_img)

        for i in range(face_num):
            y1 = faceRects[i][0]
            y2 = faceRects[i][1]
            x1 = faceRects[i][2]
            x2 = faceRects[i][3]
            pad_y1, pad_y2, pad_x1, pad_x2 = self.pad_box(
                y1, y2, x1, x2, 1.47, 1.47, 1.3, 2.0, h, w)
            temp_img = image.copy()
            for j in range(face_num):
                y1 = faceRects[j][0]
                y2 = faceRects[j][1]
                x1 = faceRects[j][2]
                x2 = faceRects[j][3]
                small_y1, small_y2, small_x1, small_x2 = self.pad_box(
                    y1, y2, x1, x2, -0.1, -0.1, -0.1, -0.1, h, w)
                small_width = small_x2 - small_x1
                small_height = small_y2 - small_y1
                if (small_x1 < 0 or small_y1 < 0 or small_width < 3
                        or small_height < 3 or small_x2 > w or small_y2 > h):
                    continue
                # if(i!=j):
                #     temp_img[small_y1:small_y2,small_x1:small_x2]=0
                if (i != j):
                    temp_img = temp_img * (1.0 - all_face_mask[j] / 255.0)

            roi_img = temp_img[pad_y1:pad_y2, pad_x1:pad_x2]
            output_alpha = self.sess.run(
                self.sess.graph.get_tensor_by_name('output_alpha:0'),
                feed_dict={'input_image:0': roi_img[:, :, ::-1]})
            head_alpha = np.zeros((h, w))
            head_alpha[pad_y1:pad_y2, pad_x1:pad_x2] = output_alpha[:, :, 0]
            if np.sum(head_alpha) > 255 * w * h * 0.01 * 0.01:
                all_head_alpha.append(head_alpha)

        head_num = len(all_head_alpha)
        head_elements = []
        if head_num == 0:
            return head_elements

        for i in range(head_num):
            head_alpha = all_head_alpha[i]
            head_elements.append(head_alpha)

        return head_elements

    def pad_box(self, y1, y2, x1, x2, left_ratio, right_ratio, top_ratio,
                bottom_ratio, h, w):
        box_w = x2 - x1
        box_h = y2 - y1
        pad_y1 = np.maximum(np.int32(y1 - top_ratio * box_h), 0)
        pad_y2 = np.minimum(np.int32(y2 + bottom_ratio * box_h), h - 1)
        pad_x1 = np.maximum(np.int32(x1 - left_ratio * box_w), 0)
        pad_x2 = np.minimum(np.int32(x2 + right_ratio * box_w), w - 1)
        return pad_y1, pad_y2, pad_x1, pad_x2

    def detect_face(self, img):
        h, w, c = img.shape
        input_img = cv2.resize(img[:, :, ::-1], (512, 512))
        boxes, scores, num_detections = self.sess_detect.run(
            [
                self.sess_detect.graph.get_tensor_by_name('tower_0/boxes:0'),
                self.sess_detect.graph.get_tensor_by_name('tower_0/scores:0'),
                self.sess_detect.graph.get_tensor_by_name(
                    'tower_0/num_detections:0')
            ],
            feed_dict={
                'tower_0/images:0': input_img[np.newaxis],
                'training_flag:0': False
            })
        faceRects = []
        for i in range(num_detections[0]):
            if scores[0, i] < 0.5:
                continue
            y1 = np.int32(boxes[0, i, 0] * h)
            x1 = np.int32(boxes[0, i, 1] * w)
            y2 = np.int32(boxes[0, i, 2] * h)
            x2 = np.int32(boxes[0, i, 3] * w)
            if x2 <= x1 + 3 or y2 <= y1 + 3:
                continue
            faceRects.append((y1, y2, x1, x2, y2 - y1, x2 - x1))
        sorted(faceRects, key=lambda x: x[4] * x[5], reverse=True)
        return faceRects

    def generate_json(self, status_code, status_msg, ori_url, result_element,
                      track_id):
        data = {}
        data['originUri'] = ori_url
        data['elements'] = result_element
        data['statusCode'] = status_code
        data['statusMessage'] = status_msg
        data['requestId'] = track_id
        return json.dumps(data)

    def get_box(self, alpha):
        h, w = alpha.shape
        start_h = 0
        end_h = 0
        start_w = 0
        end_w = 0
        for i in range(0, h, 3):
            line = alpha[i, :]
            if np.max(line) >= 1:
                start_h = i
                break

        for i in range(0, w, 3):
            line = alpha[:, i]
            if np.max(line) >= 1:
                start_w = i
                break

        for i in range(0, h, 3):
            i = h - 1 - i
            line = alpha[i, :]
            if np.max(line) >= 1:
                end_h = i
                if end_h < h - 1:
                    end_h = end_h + 1
                break
        for i in range(0, w, 3):
            i = w - 1 - i
            line = alpha[:, i]
            if np.max(line) >= 1:
                end_w = i
                if end_w < w - 1:
                    end_w = end_w + 1
                break

        return start_h, start_w, end_h, end_w
