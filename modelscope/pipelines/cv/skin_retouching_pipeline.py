# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import tensorflow as tf
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from modelscope.metainfo import Pipelines
from modelscope.models.cv.skin_retouching.detection_model.detection_unet_in import \
    DetectionUNet
from modelscope.models.cv.skin_retouching.inpainting_model.inpainting_unet import \
    RetouchingNet
from modelscope.models.cv.skin_retouching.unet_deploy import UNet
from modelscope.models.cv.skin_retouching.utils import *  # noqa F403
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors import LoadImage
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device, device_placement
from modelscope.utils.logger import get_logger

if tf.__version__ >= '2.0':
    tf = tf.compat.v1
    tf.disable_eager_execution()

logger = get_logger()


@PIPELINES.register_module(
    Tasks.skin_retouching, module_name=Pipelines.skin_retouching)
class SkinRetouchingPipeline(Pipeline):

    def __init__(self, model: str, device: str):
        """
        use `model` to create a skin retouching pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, device=device)

        device = create_device(self.device_name)
        model_path = os.path.join(self.model, ModelFile.TORCH_MODEL_FILE)
        local_model_path = os.path.join(self.model, 'joint_20210926.pth')
        skin_model_path = os.path.join(self.model, ModelFile.TF_GRAPH_FILE)

        self.generator = UNet(3, 3).to(device)
        self.generator.load_state_dict(
            torch.load(model_path, map_location='cpu')['generator'])
        self.generator.eval()

        det_model_id = 'damo/cv_resnet50_face-detection_retinaface'
        self.detector = pipeline(Tasks.face_detection, model=det_model_id)
        self.detector.detector.to(device)

        self.local_model_path = local_model_path
        ckpt_dict_load = torch.load(self.local_model_path, map_location='cpu')
        self.inpainting_net = RetouchingNet(
            in_channels=4, out_channels=3).to(device)
        self.detection_net = DetectionUNet(
            n_channels=3, n_classes=1).to(device)

        self.inpainting_net.load_state_dict(ckpt_dict_load['inpainting_net'])
        self.detection_net.load_state_dict(ckpt_dict_load['detection_net'])

        self.inpainting_net.eval()
        self.detection_net.eval()

        self.patch_size = 512

        self.skin_model_path = skin_model_path
        if self.skin_model_path is not None:
            with device_placement(self.framework, self.device_name):
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
                config.gpu_options.allow_growth = True
                self.sess = tf.Session(config=config)
                with tf.gfile.FastGFile(self.skin_model_path, 'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    self.sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    self.sess.run(tf.global_variables_initializer())

        self.image_files_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.diffuse_mask = gen_diffuse_mask()
        self.diffuse_mask = torch.from_numpy(
            self.diffuse_mask).to(device).float()
        self.diffuse_mask = self.diffuse_mask.permute(2, 0, 1)[None, ...]

        self.input_size = 512
        self.device = device

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(float)
        result = {'img': img}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = input['img'].astype(np.uint8)

        retouch_local = True
        whitening = True
        degree = 1.0
        whitening_degree = 0.8
        return_mg = False

        with torch.no_grad():
            if whitening and whitening_degree > 0 and self.skin_model_path is not None:
                rgb_image_small, resize_scale = resize_on_long_side(
                    rgb_image, 800)
                skin_mask = self.sess.run(
                    self.sess.graph.get_tensor_by_name('output_png:0'),
                    feed_dict={'input_image:0': rgb_image_small})

            output_pred = torch.from_numpy(rgb_image).to(self.device)
            if return_mg:
                output_mg = np.ones(
                    (rgb_image.shape[0], rgb_image.shape[1], 3),
                    dtype=np.float32) * 0.5

            det_results = self.detector(rgb_image)
            # list, [{'bbox':, [x1, y1, x2, y2], 'score'...}, ...]
            results = []
            for i in range(len(det_results['scores'])):
                info_dict = {}
                info_dict['bbox'] = np.array(det_results['boxes'][i]).astype(
                    np.int32).tolist()
                info_dict['score'] = det_results['scores'][i]
                info_dict['landmarks'] = np.array(
                    det_results['keypoints'][i]).astype(np.int32).reshape(
                        5, 2).tolist()
                results.append(info_dict)

            crop_bboxes = get_crop_bbox(results)

            face_num = len(crop_bboxes)
            if face_num == 0:
                output = {
                    'pred': output_pred.cpu().numpy()[:, :, ::-1],
                    'face_num': face_num
                }
                return output

            flag_bigKernal = False
            for bbox in crop_bboxes:
                roi, expand, crop_tblr = get_roi_without_padding(
                    rgb_image, bbox)
                roi = roi_to_tensor(roi)  # bgr -> rgb

                if roi.shape[2] > 0.4 * rgb_image.shape[0]:
                    flag_bigKernal = True

                roi = roi.to(self.device)

                roi = preprocess_roi(roi)

                if retouch_local and self.local_model_path is not None:
                    roi = self.retouch_local(roi)

                roi_output = self.predict_roi(
                    roi,
                    degree=degree,
                    smooth_border=True,
                    return_mg=return_mg)

                roi_pred = roi_output['pred']
                output_pred[crop_tblr[0]:crop_tblr[1],
                            crop_tblr[2]:crop_tblr[3]] = roi_pred

                if return_mg:
                    roi_mg = roi_output['pred_mg']
                    output_mg[crop_tblr[0]:crop_tblr[1],
                              crop_tblr[2]:crop_tblr[3]] = roi_mg

            if whitening and whitening_degree > 0 and self.skin_model_path is not None:
                output_pred = whiten_img(
                    output_pred,
                    skin_mask,
                    whitening_degree,
                    flag_bigKernal=flag_bigKernal)

            if not isinstance(output_pred, np.ndarray):
                output_pred = output_pred.cpu().numpy()

            output_pred = output_pred[:, :, ::-1]

            return {OutputKeys.OUTPUT_IMG: output_pred}

    def retouch_local(self, image):
        """
        image: rgb
        """
        with torch.no_grad():
            sub_H, sub_W = image.shape[2:]

            sub_image_standard = F.interpolate(
                image, size=(768, 768), mode='bilinear', align_corners=True)
            sub_mask_pred = torch.sigmoid(
                self.detection_net(sub_image_standard))
            sub_mask_pred = F.interpolate(
                sub_mask_pred, size=(sub_H, sub_W), mode='nearest')

            sub_mask_pred_hard_low = (sub_mask_pred >= 0.35).float()
            sub_mask_pred_hard_high = (sub_mask_pred >= 0.5).float()
            sub_mask_pred = sub_mask_pred * (
                1 - sub_mask_pred_hard_high) + sub_mask_pred_hard_high
            sub_mask_pred = sub_mask_pred * sub_mask_pred_hard_low
            sub_mask_pred = 1 - sub_mask_pred

            sub_H_standard = sub_H if sub_H % self.patch_size == 0 else (
                sub_H // self.patch_size + 1) * self.patch_size
            sub_W_standard = sub_W if sub_W % self.patch_size == 0 else (
                sub_W // self.patch_size + 1) * self.patch_size

            sub_image_padding = F.pad(
                image,
                pad=(0, sub_W_standard - sub_W, 0, sub_H_standard - sub_H, 0,
                     0),
                mode='constant',
                value=0)
            sub_mask_pred_padding = F.pad(
                sub_mask_pred,
                pad=(0, sub_W_standard - sub_W, 0, sub_H_standard - sub_H, 0,
                     0),
                mode='constant',
                value=0)

            sub_image_padding = patch_partition_overlap(
                sub_image_padding, p1=self.patch_size, p2=self.patch_size)
            sub_mask_pred_padding = patch_partition_overlap(
                sub_mask_pred_padding, p1=self.patch_size, p2=self.patch_size)
            B_padding, C_padding, _, _ = sub_image_padding.size()

            sub_comp_padding_list = []
            for window_item in range(B_padding):
                sub_image_padding_window = sub_image_padding[
                    window_item:window_item + 1]
                sub_mask_pred_padding_window = sub_mask_pred_padding[
                    window_item:window_item + 1]

                sub_input_image_padding_window = sub_image_padding_window * sub_mask_pred_padding_window

                sub_output_padding_window = self.inpainting_net(
                    sub_input_image_padding_window,
                    sub_mask_pred_padding_window)
                sub_comp_padding_window = sub_input_image_padding_window + (
                    1
                    - sub_mask_pred_padding_window) * sub_output_padding_window

                sub_comp_padding_list.append(sub_comp_padding_window)

            sub_comp_padding = torch.cat(sub_comp_padding_list, dim=0)
            sub_comp = patch_aggregation_overlap(
                sub_comp_padding,
                h=int(round(sub_H_standard / self.patch_size)),
                w=int(round(sub_W_standard
                            / self.patch_size)))[:, :, :sub_H, :sub_W]

            return sub_comp

    def predict_roi(self,
                    roi,
                    degree=1.0,
                    smooth_border=False,
                    return_mg=False):
        with torch.no_grad():
            image = F.interpolate(
                roi, (self.input_size, self.input_size), mode='bilinear')

            pred_mg = self.generator(image)  # value: 0~1
            pred_mg = (pred_mg - 0.5) * degree + 0.5
            pred_mg = pred_mg.clamp(0.0, 1.0)
            pred_mg = F.interpolate(pred_mg, roi.shape[2:], mode='bilinear')
            pred_mg = pred_mg[0].permute(
                1, 2, 0)  # ndarray, (h, w, 1) or (h0, w0, 3)
            if len(pred_mg.shape) == 2:
                pred_mg = pred_mg[..., None]

            if smooth_border:
                pred_mg = smooth_border_mg(self.diffuse_mask, pred_mg)

            image = (roi[0].permute(1, 2, 0) + 1.0) / 2

            pred = (1 - 2 * pred_mg
                    ) * image * image + 2 * pred_mg * image  # value: 0~1

            pred = (pred * 255.0).byte()  # ndarray, (h, w, 3), rgb

            output = {'pred': pred}
            if return_mg:
                output['pred_mg'] = pred_mg.cpu().numpy()
            return output

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
