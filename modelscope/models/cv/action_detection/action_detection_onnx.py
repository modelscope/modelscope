# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import shutil
import subprocess

import cv2
import numpy as np
import onnxruntime as rt

from modelscope.models import Model
from modelscope.utils.constant import Devices
from modelscope.utils.device import verify_device


class ActionDetONNX(Model):

    def __init__(self, model_dir, config, *args, **kwargs):
        super().__init__(self, model_dir, *args, **kwargs)
        model_file = osp.join(config['model_file'])
        device_type, device_id = verify_device(self._device_name)
        options = rt.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        if device_type == Devices.gpu:
            sess = rt.InferenceSession(
                model_file,
                providers=['CUDAExecutionProvider'],
                sess_options=options,
                provider_options=[{
                    'device_id': device_id
                }])
        else:
            sess = rt.InferenceSession(
                model_file,
                providers=['CPUExecutionProvider'],
                sess_options=options)
        self.input_name = sess.get_inputs()[0].name
        self.sess = sess
        self.num_stride = len(config['fpn_strides'])
        self.score_thresh = np.asarray(
            config['pre_nms_thresh'], dtype='float32').reshape((1, -1))
        self.size_divisibility = config['size_divisibility']
        self.nms_threshold = config['nms_thresh']
        self.tmp_dir = config['tmp_dir']
        self.temporal_stride = config['step']
        self.input_data_type = config['input_type']
        self.action_names = config['action_names']
        self.video_length_limit = config['video_length_limit']

    def resize_box(self, det, height, width, scale_h, scale_w):
        bboxs = det[0]
        bboxs[:, [0, 2]] *= scale_w
        bboxs[:, [1, 3]] *= scale_h
        bboxs[:, [0, 2]] = bboxs[:, [0, 2]].clip(0, width - 1)
        bboxs[:, [1, 3]] = bboxs[:, [1, 3]].clip(0, height - 1)
        result = {
            'boxes': bboxs.round().astype('int32').tolist(),
            'scores': det[1].tolist(),
            'labels': [self.action_names[i] for i in det[2].tolist()]
        }
        return result

    def parse_frames(self, frame_names):
        imgs = [cv2.imread(name)[:, :, ::-1] for name in frame_names]
        imgs = np.stack(imgs).astype(self.input_data_type).transpose(
            (3, 0, 1, 2))  # c,t,h,w
        imgs = imgs[None]
        return imgs

    def forward_img(self, imgs, h, w):
        pred = self.sess.run(None, {
            self.input_name: imgs,
            'height': np.asarray(h),
            'width': np.asarray(w)
        })
        dets = self.post_nms(
            pred,
            score_threshold=self.score_thresh,
            nms_threshold=self.nms_threshold)
        return dets

    def forward_video(self, video_name, scale):
        min_size, max_size = self._get_sizes(scale)

        tmp_dir = osp.join(self.tmp_dir, osp.basename(video_name)[:-4])
        if osp.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)
        frame_rate = 2
        cmd = f'ffmpeg -y -loglevel quiet -ss 0 -t {self.video_length_limit}' + \
              f' -i {video_name} -r {frame_rate} -f image2 {tmp_dir}/%06d.jpg'

        cmd = cmd.split(' ')
        subprocess.call(cmd)

        frame_names = [
            osp.join(tmp_dir, name) for name in sorted(os.listdir(tmp_dir))
            if name.endswith('.jpg')
        ]
        frame_names = [
            frame_names[i:i + frame_rate * 2]
            for i in range(0,
                           len(frame_names) - frame_rate * 2 + 1, frame_rate
                           * self.temporal_stride)
        ]
        timestamp = list(
            range(1,
                  len(frame_names) * self.temporal_stride,
                  self.temporal_stride))
        batch_imgs = [self.parse_frames(names) for names in frame_names]

        N, _, T, H, W = batch_imgs[0].shape
        scale_min = min_size / min(H, W)
        h, w = min(int(scale_min * H),
                   max_size), min(int(scale_min * W), max_size)
        h = round(h / self.size_divisibility) * self.size_divisibility
        w = round(w / self.size_divisibility) * self.size_divisibility
        scale_h, scale_w = H / h, W / w

        results = []
        for imgs in batch_imgs:
            det = self.forward_img(imgs, h, w)
            det = self.resize_box(det[0], H, W, scale_h, scale_w)
            results.append(det)
        results = [{
            'timestamp': t,
            'actions': res
        } for t, res in zip(timestamp, results)]
        shutil.rmtree(tmp_dir)
        return results

    def forward(self, video_name):
        return self.forward_video(video_name, scale=1)

    def post_nms(self, pred, score_threshold, nms_threshold=0.3):
        pred_bboxes, pred_scores = pred
        N = len(pred_bboxes)
        dets = []
        for i in range(N):
            bboxes, scores = pred_bboxes[i], pred_scores[i]
            candidate_inds = scores > score_threshold
            scores = scores[candidate_inds]
            candidate_nonzeros = candidate_inds.nonzero()
            bboxes = bboxes[candidate_nonzeros[0]]
            labels = candidate_nonzeros[1]
            keep = self._nms(bboxes, scores, labels, nms_threshold)
            bbox = bboxes[keep]
            score = scores[keep]
            label = labels[keep]
            dets.append((bbox, score, label))
        return dets

    def _nms(self, boxes, scores, idxs, nms_threshold):
        if len(boxes) == 0:
            return []
        max_coordinate = boxes.max()
        offsets = idxs * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None].astype('float32')
        boxes_for_nms[:, 2] = boxes_for_nms[:, 2] - boxes_for_nms[:, 0]
        boxes_for_nms[:, 3] = boxes_for_nms[:, 3] - boxes_for_nms[:, 1]
        keep = cv2.dnn.NMSBoxes(
            boxes_for_nms.tolist(),
            scores.tolist(),
            score_threshold=0,
            nms_threshold=nms_threshold)
        if len(keep.shape) == 2:
            keep = np.squeeze(keep, 1)
        return keep

    def _get_sizes(self, scale):
        if scale == 1:
            min_size, max_size = 512, 896
        elif scale == 2:
            min_size, max_size = 768, 1280
        else:
            min_size, max_size = 1024, 1792
        return min_size, max_size
