# Copyright (c) Alibaba, Inc. and its affiliates.
import math
import os
from collections import OrderedDict
from typing import Any, Dict

import cv2
import json
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from scipy.io import loadmat

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.face_detection.peppa_pig_face.facer import FaceAna
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from .network.camera_utils import FOV_to_intrinsics, LookAtPoseSampler
from .network.shape_utils import convert_sdf_samples_to_ply
from .network.triplane import TriPlaneGenerator
from .network.triplane_encoder import TriplaneEncoder

logger = get_logger()

__all__ = ['ImageControl3dPortrait']


@MODELS.register_module(
    Tasks.image_control_3d_portrait,
    module_name=Models.image_control_3d_portrait)
class ImageControl3dPortrait(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the image face fusion model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        logger.info('model params:{}'.format(kwargs))
        self.neural_rendering_resolution = kwargs[
            'neural_rendering_resolution']
        self.cam_radius = kwargs['cam_radius']
        self.fov_deg = kwargs['fov_deg']
        self.truncation_psi = kwargs['truncation_psi']
        self.truncation_cutoff = kwargs['truncation_cutoff']
        self.z_dim = kwargs['z_dim']
        self.image_size = kwargs['image_size']
        self.shape_res = kwargs['shape_res']
        self.pitch_range = kwargs['pitch_range']
        self.yaw_range = kwargs['yaw_range']
        self.max_batch = kwargs['max_batch']
        self.num_frames = kwargs['num_frames']
        self.box_warp = kwargs['box_warp']
        self.save_shape = kwargs['save_shape']
        self.save_images = kwargs['save_images']

        device = kwargs['device']
        self.device = create_device(device)

        self.facer = FaceAna(model_dir)

        similarity_mat_path = os.path.join(model_dir, 'BFM',
                                           'similarity_Lm3D_all.mat')
        self.lm3d_std = self.load_lm3d(similarity_mat_path)

        init_model_json = os.path.join(model_dir, 'configs',
                                       'init_encoder.json')
        with open(init_model_json, 'r') as fr:
            init_kwargs_encoder = json.load(fr)
        encoder_path = os.path.join(model_dir, ModelFile.TORCH_MODEL_FILE)
        self.model = TriplaneEncoder(**init_kwargs_encoder)
        ckpt_encoder = torch.load(encoder_path, map_location='cpu')
        model_state = self.convert_state_dict(ckpt_encoder['state_dict'])
        self.model.load_state_dict(model_state)
        self.model = self.model.to(self.device)
        self.model.eval()

        init_args_G = ()
        init_netG_json = os.path.join(model_dir, 'configs', 'init_G.json')
        with open(init_netG_json, 'r') as fr:
            init_kwargs_G = json.load(fr)
        self.netG = TriPlaneGenerator(*init_args_G, **init_kwargs_G)
        netG_path = os.path.join(model_dir, 'ffhqrebalanced512-128.pth')
        ckpt_G = torch.load(netG_path)
        self.netG.load_state_dict(ckpt_G['G_ema'], strict=False)
        self.netG.neural_rendering_resolution = self.neural_rendering_resolution
        self.netG = self.netG.to(self.device)
        self.netG.eval()

        self.intrinsics = FOV_to_intrinsics(self.fov_deg, device=self.device)
        col, row = np.meshgrid(
            np.arange(self.image_size), np.arange(self.image_size))
        np_coord = np.stack((col, row), axis=2) / self.image_size  # [0,1]
        self.coord = torch.from_numpy(np_coord.astype(
            np.float32)).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        logger.info('init done')

    def convert_state_dict(self, state_dict):
        if not next(iter(state_dict)).startswith('module.'):
            return state_dict
        new_state_dict = OrderedDict()

        split_index = 0
        for cur_key, cur_value in state_dict.items():
            if cur_key.startswith('module.model'):
                split_index = 13
            elif cur_key.startswith('module'):
                split_index = 7

            break

        for k, v in state_dict.items():
            name = k[split_index:]
            new_state_dict[name] = v
        return new_state_dict

    def detect_face(self, img):
        src_h, src_w, _ = img.shape
        boxes, landmarks, _ = self.facer.run(img)
        if boxes.shape[0] == 0:
            return None
        elif boxes.shape[0] > 1:
            max_area = 0
            max_index = 0
            for i in range(boxes.shape[0]):
                bbox_width = boxes[i][2] - boxes[i][0]
                bbox_height = boxes[i][3] - boxes[i][1]
                area = int(bbox_width) * int(bbox_height)
                if area > max_area:
                    max_index = i
                    max_area = area

            return landmarks[max_index]
        else:
            return landmarks[0]

    def get_f5p(self, landmarks, np_img):
        eye_left = self.find_pupil(landmarks[36:41], np_img)
        eye_right = self.find_pupil(landmarks[42:47], np_img)
        if eye_left is None or eye_right is None:
            logger.warning(
                'cannot find 5 points with find_pupil, used mean instead.!')
            eye_left = landmarks[36:41].mean(axis=0)
            eye_right = landmarks[42:47].mean(axis=0)
        nose = landmarks[30]
        mouth_left = landmarks[48]
        mouth_right = landmarks[54]
        f5p = [[eye_left[0], eye_left[1]], [eye_right[0], eye_right[1]],
               [nose[0], nose[1]], [mouth_left[0], mouth_left[1]],
               [mouth_right[0], mouth_right[1]]]
        return np.array(f5p)

    def find_pupil(self, landmarks, np_img):
        h, w, _ = np_img.shape
        xmax = int(landmarks[:, 0].max())
        xmin = int(landmarks[:, 0].min())
        ymax = int(landmarks[:, 1].max())
        ymin = int(landmarks[:, 1].min())

        if ymin >= ymax or xmin >= xmax or ymin < 0 or xmin < 0 or ymax > h or xmax > w:
            return None
        eye_img_bgr = np_img[ymin:ymax, xmin:xmax, :]
        eye_img = cv2.cvtColor(eye_img_bgr, cv2.COLOR_BGR2GRAY)
        eye_img = cv2.equalizeHist(eye_img)
        n_marks = landmarks - np.array([xmin, ymin]).reshape([1, 2])
        eye_mask = cv2.fillConvexPoly(
            np.zeros_like(eye_img), n_marks.astype(np.int32), 1)
        ret, thresh = cv2.threshold(eye_img, 100, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = (1 - thresh / 255.) * eye_mask
        cnt = 0
        xm = []
        ym = []
        for i in range(thresh.shape[0]):
            for j in range(thresh.shape[1]):
                if thresh[i, j] > 0.5:
                    xm.append(j)
                    ym.append(i)
                    cnt += 1
        if cnt != 0:
            xm.sort()
            ym.sort()
            xm = xm[cnt // 2]
            ym = ym[cnt // 2]
        else:
            xm = thresh.shape[1] / 2
            ym = thresh.shape[0] / 2

        return xm + xmin, ym + ymin

    def load_lm3d(self, similarity_mat_path):

        Lm3D = loadmat(similarity_mat_path)
        Lm3D = Lm3D['lm']

        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm_data1 = Lm3D[lm_idx[0], :]
        lm_data2 = np.mean(Lm3D[lm_idx[[1, 2]], :], 0)
        lm_data3 = np.mean(Lm3D[lm_idx[[3, 4]], :], 0)
        lm_data4 = Lm3D[lm_idx[5], :]
        lm_data5 = Lm3D[lm_idx[6], :]

        Lm3D = np.stack([lm_data1, lm_data2, lm_data3, lm_data4, lm_data5],
                        axis=0)

        Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

        return Lm3D

    def POS(self, xp, x):
        npts = xp.shape[1]

        A = np.zeros([2 * npts, 8])

        A[0:2 * npts - 1:2, 0:3] = x.transpose()
        A[0:2 * npts - 1:2, 3] = 1

        A[1:2 * npts:2, 4:7] = x.transpose()
        A[1:2 * npts:2, 7] = 1

        b = np.reshape(xp.transpose(), [2 * npts, 1])

        k, _, _, _ = np.linalg.lstsq(A, b)

        R1 = k[0:3]
        R2 = k[4:7]
        sTx = k[3]
        sTy = k[7]
        s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2
        t = np.stack([sTx, sTy], axis=0)

        return t, s

    def resize_n_crop_img(self, img, lm, t, s, target_size=224., mask=None):
        w0, h0 = img.size
        w = (w0 * s).astype(np.int32)
        h = (h0 * s).astype(np.int32)
        left = (w / 2 - target_size / 2 + float(
            (t[0] - w0 / 2) * s)).astype(np.int32)
        right = left + target_size
        up = (h / 2 - target_size / 2 + float(
            (h0 / 2 - t[1]) * s)).astype(np.int32)
        below = up + target_size

        img = img.resize((w, h), resample=Image.BICUBIC)
        img = img.crop((left, up, right, below))

        if mask is not None:
            mask = mask.resize((w, h), resample=Image.BICUBIC)
            mask = mask.crop((left, up, right, below))

        lm = np.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2],
                      axis=1) * s
        lm = lm - np.reshape(
            np.array([(w / 2 - target_size / 2),
                      (h / 2 - target_size / 2)]), [1, 2])

        return img, lm, mask

    def align_img(self,
                  img,
                  lm,
                  lm3D,
                  mask=None,
                  target_size=224.,
                  rescale_factor=102.):
        w0, h0 = img.size
        lm5p = lm
        t, s = self.POS(lm5p.transpose(), lm3D.transpose())
        s = rescale_factor / s

        img_new, lm_new, mask_new = self.resize_n_crop_img(
            img, lm, t, s, target_size=target_size, mask=mask)
        trans_params = np.array([w0, h0, s, t[0], t[1]], dtype=object)

        return trans_params, img_new, lm_new, mask_new

    def crop_image(self, img, lm):
        _, H = img.size
        lm[:, -1] = H - 1 - lm[:, -1]

        target_size = 1024.
        rescale_factor = 300
        center_crop_size = 700
        output_size = 512

        _, im_high, _, _, = self.align_img(
            img,
            lm,
            self.lm3d_std,
            target_size=target_size,
            rescale_factor=rescale_factor)

        left = int(im_high.size[0] / 2 - center_crop_size / 2)
        upper = int(im_high.size[1] / 2 - center_crop_size / 2)
        right = left + center_crop_size
        lower = upper + center_crop_size
        im_cropped = im_high.crop((left, upper, right, lower))
        im_cropped = im_cropped.resize((output_size, output_size),
                                       resample=Image.LANCZOS)
        logger.info('crop image done!')
        return im_cropped

    def create_samples(self, N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
        voxel_origin = np.array(voxel_origin) - cube_length / 2
        voxel_size = cube_length / (N - 1)

        overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
        samples = torch.zeros(N**3, 3)

        samples[:, 2] = overall_index % N
        samples[:, 1] = (overall_index.float() / N) % N
        samples[:, 0] = ((overall_index.float() / N) / N) % N

        samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
        samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
        samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

        return samples.unsqueeze(0), voxel_origin, voxel_size

    def numpy_array_to_video(self, numpy_list, video_out_path):
        assert len(numpy_list) > 0
        video_height = numpy_list[0].shape[0]
        video_width = numpy_list[0].shape[1]

        out_video_size = (video_width, video_height)
        output_video_fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        video_write_capture = cv2.VideoWriter(video_out_path,
                                              output_video_fourcc, 30,
                                              out_video_size)

        for frame in numpy_list:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_write_capture.write(frame_bgr)

        video_write_capture.release()

    def inference(self, image_path, save_dir):
        basename = os.path.basename(image_path).split('.')[0]
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        img_bgr = img_array[:, :, ::-1]
        landmark = self.detect_face(img_array)
        if landmark is None:
            logger.warning('No face detected in the image!')
        f5p = self.get_f5p(landmark, img_bgr)

        logger.info('f5p is:{}'.format(f5p))
        img_cropped = self.crop_image(img, f5p)
        img_cropped.save(os.path.join(save_dir, 'crop.jpg'))

        in_image = self.image_transform(img_cropped).unsqueeze(0).to(
            self.device)
        input = torch.cat((in_image, self.coord), 1)

        save_video_path = os.path.join(save_dir, f'{basename}.mp4')
        pred_imgs = []

        for frame_idx in range(self.num_frames):
            cam_pivot = torch.tensor([0, 0, 0.2], device=self.device)

            cam2world_pose = LookAtPoseSampler.sample(
                3.14 / 2 + self.yaw_range
                * np.sin(2 * 3.14 * frame_idx / self.num_frames),
                3.14 / 2 - 0.05 + self.pitch_range
                * np.cos(2 * 3.14 * frame_idx / self.num_frames),
                cam_pivot,
                radius=self.cam_radius,
                device=self.device)

            camera_params = torch.cat([
                cam2world_pose.reshape(-1, 16),
                self.intrinsics.reshape(-1, 9)
            ], 1)

            conditioning_cam2world_pose = LookAtPoseSampler.sample(
                np.pi / 2,
                np.pi / 2,
                cam_pivot,
                radius=self.cam_radius,
                device=self.device)
            conditioning_params = torch.cat([
                conditioning_cam2world_pose.reshape(-1, 16),
                self.intrinsics.reshape(-1, 9)
            ], 1)

            z = torch.from_numpy(np.random.randn(1,
                                                 self.z_dim)).to(self.device)

            with torch.no_grad():
                ws = self.netG.mapping(
                    z,
                    conditioning_params,
                    truncation_psi=self.truncation_psi,
                    truncation_cutoff=self.truncation_cutoff)

                planes, pred_depth, pred_feature, pred_rgb, pred_sr, _, _, _, _ = self.model(
                    ws, input, camera_params, None)

            pred_img = (pred_sr.permute(0, 2, 3, 1) * 127.5 + 128).clamp(
                0, 255).to(torch.uint8)
            pred_img = pred_img.squeeze().cpu().numpy()
            if self.save_images:
                cv2.imwrite(
                    os.path.join(save_dir, '{}.jpg'.format(frame_idx)),
                    pred_img[:, :, ::-1])
            pred_imgs.append(pred_img)

        self.numpy_array_to_video(pred_imgs, save_video_path)

        if self.save_shape:
            max_batch = 1000000

            samples, voxel_origin, voxel_size = self.create_samples(
                N=self.shape_res,
                voxel_origin=[0, 0, 0],
                cube_length=self.box_warp)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1),
                                 device=z.device)
            transformed_ray_directions_expanded = torch.zeros(
                (samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with torch.no_grad():
                while head < samples.shape[1]:
                    torch.manual_seed(0)
                    sigma = self.model.sample(
                        samples[:, head:head + max_batch],
                        transformed_ray_directions_expanded[:, :samples.
                                                            shape[1] - head],
                        planes)['sigma']
                    sigmas[:, head:head + max_batch] = sigma
                    head += max_batch

            sigmas = sigmas.reshape((self.shape_res, self.shape_res,
                                     self.shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            pad = int(30 * self.shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value
            convert_sdf_samples_to_ply(
                np.transpose(sigmas, (2, 1, 0)), [0, 0, 0],
                1,
                os.path.join(save_dir, f'{basename}.ply'),
                level=10)

        logger.info('model inference done')
