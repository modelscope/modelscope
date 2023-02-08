# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
from typing import Any, Dict

import cv2
import face_alignment
import numpy as np
import PIL.Image
import tensorflow as tf
import torch
from scipy.io import loadmat, savemat

from modelscope.metainfo import Pipelines
from modelscope.models import Model
from modelscope.models.cv.face_reconstruction.models.facelandmark.large_model_infer import \
    LargeModelInfer
from modelscope.models.cv.face_reconstruction.utils import (align_for_lm,
                                                            align_img,
                                                            load_lm3d,
                                                            read_obj,
                                                            write_obj)
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
    Tasks.face_reconstruction, module_name=Pipelines.face_reconstruction)
class FaceReconstructionPipeline(Pipeline):

    def __init__(self, model: str, device: str):
        """The inference pipeline for face reconstruction task.

        Args:
            model (`str` or `Model` or module instance): A model instance or a model local dir
                or a model id in the model hub.
            device ('str'): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X.

        Example:
            >>> from modelscope.pipelines import pipeline
            >>> test_image = 'data/test/images/face_reconstruction.jpg'
            >>> pipeline_faceRecon = pipeline('face-reconstruction',
                model='damo/cv_resnet50_face-reconstruction')
            >>> result = pipeline_faceRecon(test_image)
            >>> write_obj('result_face_reconstruction.obj', result[OutputKeys.OUTPUT])
        """
        super().__init__(model=model, device=device)

        model_root = model
        bfm_folder = os.path.join(model_root, 'assets')
        checkpoint_path = os.path.join(model_root, ModelFile.TORCH_MODEL_FILE)

        self.face_mark_model = LargeModelInfer(
            os.path.join(model_root, 'large_base_net.pth'), device='cuda')

        device = torch.device(0)
        torch.cuda.set_device(device)
        self.model.setup(checkpoint_path)
        self.model.device = device
        self.model.parallelize()
        self.model.eval()
        self.model.set_render(image_res=1024)

        save_ckpt_dir = os.path.join(
            os.path.expanduser('~'), '.cache/torch/hub/checkpoints')
        if not os.path.exists(save_ckpt_dir):
            os.makedirs(save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment', 's3fd-619a316812.pth'),
            save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment',
                         '3DFAN4-4a694010b9.zip'), save_ckpt_dir)
        shutil.copy(
            os.path.join(model_root, 'face_alignment', 'depth-6c4283c0e0.zip'),
            save_ckpt_dir)
        self.lm_sess = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._3D, flip_input=False)

        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        config.gpu_options.allow_growth = True
        g1 = tf.Graph()
        self.face_sess = tf.Session(graph=g1, config=config)
        with self.face_sess.as_default():
            with g1.as_default():
                with tf.gfile.FastGFile(
                        os.path.join(model_root, 'segment_face.pb'),
                        'rb') as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    self.face_sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')
                    self.face_sess.run(tf.global_variables_initializer())

        self.tex_size = 4096

        self.bald_tex_bg = cv2.imread(
            '{}/assets/template_texture.jpg'.format(model_root)).astype(
                np.float32)

        front_mask = cv2.imread(
            '{}/assets/face_mask.jpg'.format(model_root)).astype(
                np.float32) / 255
        front_mask = cv2.resize(front_mask, (1024, 1024))
        front_mask = cv2.resize(front_mask, (0, 0), fx=0.1, fy=0.1)
        front_mask = cv2.erode(front_mask,
                               np.ones(shape=(7, 7), dtype=np.float32))
        front_mask = cv2.GaussianBlur(front_mask, (13, 13), 0)
        self.front_mask = cv2.resize(front_mask,
                                     (self.tex_size, self.tex_size))
        self.binary_front_mask = self.front_mask.copy()
        self.binary_front_mask[(self.front_mask < 0.3)
                               + (self.front_mask > 0.7)] = 0
        self.binary_front_mask[self.binary_front_mask != 0] = 1.0
        self.binary_front_mask_ = self.binary_front_mask.copy()
        self.binary_front_mask = np.zeros((4096 + 1024, 4096, 3),
                                          dtype=np.float32)
        self.binary_front_mask[:4096, :] = self.binary_front_mask_
        self.front_mask_ = self.front_mask.copy()
        self.front_mask = np.zeros((4096 + 1024, 4096, 3), dtype=np.float32)
        self.front_mask[:4096, :] = self.front_mask_

        l_eye_mask = cv2.imread(
            '{}/assets/l_eye_mask.png'.format(model_root))[:, :, :1] / 255.0
        l_eye_mask = cv2.erode(l_eye_mask,
                               np.ones(shape=(5, 5), dtype=np.float32))
        self.l_eye_mask = cv2.GaussianBlur(l_eye_mask, (7, 7), 0)[..., None]
        self.l_eye_binary_mask = self.l_eye_mask.copy()
        self.l_eye_binary_mask[(self.l_eye_mask < 0.3)
                               + (self.l_eye_mask > 0.7)] = 0
        self.l_eye_binary_mask[self.l_eye_binary_mask != 0] = 1.0

        r_eye_mask = cv2.imread(
            '{}/assets/r_eye_mask.png'.format(model_root))[:, :, :1] / 255.0
        r_eye_mask = cv2.dilate(r_eye_mask,
                                np.ones(shape=(7, 7), dtype=np.float32))
        self.r_eye_mask = cv2.GaussianBlur(r_eye_mask, (7, 7), 0)[..., None]
        self.r_eye_binary_mask = self.r_eye_mask.copy()
        self.r_eye_binary_mask[(self.r_eye_mask < 0.3)
                               + (self.r_eye_mask > 0.7)] = 0
        self.r_eye_binary_mask[self.r_eye_binary_mask != 0] = 1.0

        self.lm3d_std = load_lm3d(bfm_folder)
        self.align_params = loadmat(
            '{}/assets/BBRegressorParam_r.mat'.format(model_root))

        device = create_device(self.device_name)
        self.device = device

    def preprocess(self, input: Input) -> Dict[str, Any]:
        img = LoadImage.convert_to_ndarray(input)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = img.astype(np.float)
        result = {'img': img}
        return result

    def read_data(self,
                  img,
                  lm,
                  lm3d_std,
                  to_tensor=True,
                  image_res=1024,
                  img_fat=None):
        # to RGB
        im = PIL.Image.fromarray(img[..., ::-1])
        W, H = im.size
        lm[:, -1] = H - 1 - lm[:, -1]

        im_lr_coeff, lm_lr_coeff = None, None
        head_mask = None

        _, im_lr, lm_lr, mask_lr_head = align_img(
            im, lm, lm3d_std, mask=head_mask)
        _, im_hd, lm_hd, _ = align_img(
            im,
            lm,
            lm3d_std,
            target_size=image_res,
            rescale_factor=102.0 * image_res / 224)

        mask_lr = self.face_sess.run(
            self.face_sess.graph.get_tensor_by_name('output_alpha:0'),
            feed_dict={'input_image:0': np.array(im_lr)})

        if img_fat is not None:
            assert img_fat.shape == img.shape
            im_fat = PIL.Image.fromarray(img_fat[..., ::-1])

            _, im_hd, _, _ = align_img(
                im_fat,
                lm,
                lm3d_std,
                target_size=image_res,
                rescale_factor=102.0 * image_res / 224)

        im_hd = np.array(im_hd).astype(np.float32)

        if to_tensor:
            im_lr = torch.tensor(
                np.array(im_lr) / 255.,
                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            im_hd = torch.tensor(
                np.array(im_hd) / 255.,
                dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            mask_lr = torch.tensor(
                np.array(mask_lr) / 255., dtype=torch.float32)[None,
                                                               None, :, :]
            mask_lr_head = torch.tensor(
                np.array(mask_lr_head) / 255., dtype=torch.float32)[
                    None, None, :, :] if mask_lr_head is not None else None
            lm_lr = torch.tensor(lm_lr).unsqueeze(0)
            lm_hd = torch.tensor(lm_hd).unsqueeze(0)
        return im_lr, lm_lr, im_hd, lm_hd, mask_lr, mask_lr_head, im_lr_coeff, lm_lr_coeff

    def prepare_data(self, img, lm_sess, five_points=None):
        input_img, scale, bbox = align_for_lm(
            img, five_points,
            self.align_params)  # align for 68 landmark detection

        if scale == 0:
            return None

        # detect landmarks
        input_img = np.reshape(input_img, [1, 224, 224, 3]).astype(np.float32)

        input_img = input_img[0, :, :, ::-1]
        landmark = lm_sess.get_landmarks_from_image(input_img)[0]

        landmark = landmark[:, :2] / scale
        landmark[:, 0] = landmark[:, 0] + bbox[0]
        landmark[:, 1] = landmark[:, 1] + bbox[1]

        return landmark

    def blend_eye_corner(self, tex_map, template_tex):
        tex_map = tex_map.astype(np.float32)

        x1 = int(288 * 4096 / 758)
        y1 = int(235 * 4096 / 758)
        w = int(90 * 4096 / 758)
        h = int(50 * 4096 / 758)
        template_tex_l = template_tex[y1:y1 + h, x1:x1 + w]
        pred_tex_l = tex_map[y1:y1 + h, x1:x1 + w]
        pred_tex_l_mean_rgb = np.sum(
            pred_tex_l * self.l_eye_binary_mask, axis=(0, 1))
        template_tex_l_mean_rgb = np.sum(
            template_tex_l * self.l_eye_binary_mask, axis=(0, 1))
        for ch in range(3):
            template_tex_l[:, :, ch] *= pred_tex_l_mean_rgb[
                ch] / template_tex_l_mean_rgb[ch]
        pred_tex_l = pred_tex_l * (
            1 - self.l_eye_mask) + template_tex_l * self.l_eye_mask

        x2 = 4096 - x1 - w
        y2 = y1
        template_tex_r = template_tex[y2:y2 + h, x2:x2 + w]
        pred_tex_r = tex_map[y2:y2 + h, x2:x2 + w]
        pred_tex_r_mean_rgb = np.sum(
            pred_tex_r * self.r_eye_binary_mask, axis=(0, 1))
        template_tex_r_mean_rgb = np.sum(
            template_tex_r * self.r_eye_binary_mask, axis=(0, 1))
        for ch in range(3):
            template_tex_r[:, :, ch] *= pred_tex_r_mean_rgb[
                ch] / template_tex_r_mean_rgb[ch]
        pred_tex_r = pred_tex_r * (
            1 - self.r_eye_mask) + template_tex_r * self.r_eye_mask

        tex_map[y1:y1 + h, x1:x1 + w] = pred_tex_l
        tex_map[y2:y2 + h, x2:x2 + w] = pred_tex_r

        return tex_map

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        rgb_image = input['img'].cpu().numpy().astype(np.uint8)

        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

        img = bgr_image
        # preprocess
        flag = 0
        box, results = self.face_mark_model.infer(img)
        if results is None or np.array(results).shape[0] == 0:
            flag = 1  # no face
            return flag, {}

        fatbgr = self.face_mark_model.fat_face(img, degree=0.02)

        landmarks = []
        results = results[0]
        for idx in [74, 83, 54, 84, 90]:
            landmarks.append([results[idx][0], results[idx][1]])
        landmarks = np.array(landmarks)

        landmarks = self.prepare_data(img, self.lm_sess, five_points=landmarks)

        im_tensor, lm_tensor, im_hd_tensor, lm_hd_tensor, mask, _, _, _ = self.read_data(
            img, landmarks, self.lm3d_std, image_res=1024, img_fat=fatbgr)
        data = {
            'imgs': im_tensor,
            'imgs_hd': im_hd_tensor,
            'lms': lm_tensor,
            'lms_hd': lm_hd_tensor,
            'face_mask': mask,
            'img_name': 'temp',
        }
        self.model.set_input(data)  # unpack data from data loader

        # reconstruct
        out_dir = None
        output = self.model(out_dir=out_dir)  # run inference

        # process texture map
        tex_map = output['head_tex_map'].astype(np.float32)
        tex_map = cv2.resize(tex_map, (self.tex_size, self.tex_size + 1024))
        bg_mean_rgb = np.sum(
            self.bald_tex_bg * self.binary_front_mask, axis=(0, 1))
        pred_tex_mean_rgb = np.sum(
            tex_map * self.binary_front_mask, axis=(0, 1)) * 1.05
        mid_mean_rgb = bg_mean_rgb * 0.8 + pred_tex_mean_rgb * 0.2
        tex_map += (
            (mid_mean_rgb - pred_tex_mean_rgb)
            / np.sum(self.binary_front_mask, axis=(0, 1)))[None, None] * 0.5
        pred_tex_mean_rgb = np.sum(
            tex_map * self.binary_front_mask, axis=(0, 1)) * 1.05
        _bald_tex_bg = self.bald_tex_bg.copy()
        for ch in range(3):
            _bald_tex_bg[:, :, ch] *= pred_tex_mean_rgb[ch] / bg_mean_rgb[ch]
        tex_map = _bald_tex_bg * (
            1. - self.front_mask) + tex_map * self.front_mask
        tex_map = tex_map * 1.05
        tex_map = self.blend_eye_corner(tex_map, self.bald_tex_bg)

        # export mesh
        results = {
            'vertices': output['head_vertices'],
            'faces': output['head_faces'],
            'UVs': output['head_UVs'],
            'faces_uv': output['head_faces_uv'],
            'normals': output['head_normals'],
            'texture_map': tex_map,
        }

        if out_dir is not None:
            face_mesh = {
                'vertices': output['face_vertices'],
                'faces': output['face_faces'],
                'colors': output['face_colors'],
            }

            write_obj(os.path.join(out_dir, 'face.obj'), face_mesh)
            write_obj(os.path.join(out_dir, 'head.obj'), results)

        return {OutputKeys.OUTPUT: results}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs
