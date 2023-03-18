# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Optional

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torchvision.transforms as transforms
from skimage.io import imread
from skimage.transform import estimate_transform, warp

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.human_reconstruction.models.detectors import \
    FasterRCNN
from modelscope.models.cv.human_reconstruction.models.human_segmenter import \
    human_segmenter
from modelscope.models.cv.human_reconstruction.models.networks import define_G
from modelscope.models.cv.human_reconstruction.models.PixToMesh import \
    Pixto3DNet
from modelscope.models.cv.human_reconstruction.utils import create_grid
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@MODELS.register_module(
    Tasks.human_reconstruction, module_name=Models.human_reconstruction)
class HumanReconstruction(TorchModel):

    def __init__(self, model_dir, modelconfig, *args, **kwargs):
        """The HumanReconstruction is modified based on PiFuHD and pix2pixhd, publicly available at
                https://shunsukesaito.github.io/PIFuHD/ &
                https://github.com/NVIDIA/pix2pixHD

        Args:
            model_dir: the root directory of the model files
            modelconfig: the config param path of the model
        """
        super().__init__(model_dir=model_dir, *args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            logger.info('Use GPU: {}'.format(self.device))
        else:
            self.device = torch.device('cpu')
            logger.info('Use CPU: {}'.format(self.device))

        model_path = '{}/{}'.format(model_dir, ModelFile.TORCH_MODEL_FILE)
        normal_back_model = '{}/{}'.format(model_dir, 'Norm_B_GAN.pth')
        normal_front_model = '{}/{}'.format(model_dir, 'Norm_F_GAN.pth')
        human_seg_model = '{}/{}'.format(model_dir, ModelFile.TF_GRAPH_FILE)
        fastrcnn_ckpt = '{}/{}'.format(model_dir, 'fasterrcnn_resnet50.pth')
        self.meshmodel = Pixto3DNet(**modelconfig['model'])
        self.detector = FasterRCNN(ckpt=fastrcnn_ckpt, device=self.device)
        self.meshmodel.load_state_dict(
            torch.load(model_path, map_location='cpu'))
        self.netB = define_G(3, 3, 64, 'global', 4, 9, 1, 3, 'instance')
        self.netF = define_G(3, 3, 64, 'global', 4, 9, 1, 3, 'instance')
        self.netF.load_state_dict(torch.load(normal_front_model))
        self.netB.load_state_dict(torch.load(normal_back_model))
        self.netF = self.netF.to(self.device)
        self.netB = self.netB.to(self.device)
        self.netF.eval()
        self.netB.eval()
        self.meshmodel = self.meshmodel.to(self.device).eval()
        self.portrait_matting = human_segmenter(model_path=human_seg_model)
        b_min = np.array([-1, -1, -1])
        b_max = np.array([1, 1, 1])
        self.coords, self.mat = create_grid(modelconfig['resolution'], b_min,
                                            b_max)
        projection_matrix = np.identity(4)
        projection_matrix[1, 1] = -1
        self.calib = torch.Tensor(projection_matrix).float().to(self.device)
        self.calib = self.calib[:3, :4].unsqueeze(0)
        logger.info('model load over')

    def get_mask(self, img):
        result = self.portrait_matting.run(img)
        result = result[..., None]
        mask = result.repeat(3, axis=2)
        return img, mask

    @torch.no_grad()
    def crop_img(self, img_url):
        image = imread(img_url)[:, :, :3] / 255.
        h, w, _ = image.shape
        image_size = 512
        image_tensor = torch.tensor(
            image.transpose(2, 0, 1), dtype=torch.float32)[None, ...]
        bbox = self.detector.run(image_tensor)
        left = bbox[0]
        right = bbox[2]
        top = bbox[1]
        bottom = bbox[3]

        old_size = max(right - left, bottom - top)
        center = np.array(
            [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * 1.1)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2],
                            [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(
            image, tform.inverse, output_shape=(image_size, image_size))
        dst_image = (dst_image[:, :, ::-1] * 255).astype(np.uint8)
        return dst_image

    @torch.no_grad()
    def generation_normal(self, img, mask):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        im_512 = cv2.resize(img, (512, 512))
        image_512 = Image.fromarray(im_512).convert('RGB')
        image_512 = to_tensor(image_512).unsqueeze(0)
        img = image_512.to(self.device)
        nml_f = self.netF.forward(img)
        nml_b = self.netB.forward(img)
        mask = cv2.resize(mask, (512, 512))
        mask = transforms.ToTensor()(mask).unsqueeze(0)
        nml_f = (nml_f.cpu() * mask).detach().cpu().numpy()[0]
        nml_f = (np.transpose(nml_f,
                              (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        nml_b = (nml_b.cpu() * mask).detach().cpu().numpy()[0]
        nml_b = (np.transpose(nml_b,
                              (1, 2, 0)) * 0.5 + 0.5)[:, :, ::-1] * 255.0
        nml_f = nml_f.astype(np.uint8)
        nml_b = nml_b.astype(np.uint8)
        return nml_f, nml_b

    # def forward(self, img, mask, normal_f, normal_b):
