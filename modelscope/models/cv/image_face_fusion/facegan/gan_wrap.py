# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .model import FullGenerator


class GANWrap(object):

    def __init__(self,
                 model_path,
                 size=256,
                 channel_multiplier=1,
                 device='cpu'):
        self.device = device
        self.mfile = model_path
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
        ])
        self.batchSize = 2
        self.n_mlp = 8
        self.resolution = size
        self.load_model(channel_multiplier)

    def load_model(self, channel_multiplier=2):
        self.model = FullGenerator(self.resolution, 512, self.n_mlp,
                                   channel_multiplier).to(self.device)
        pretrained_dict = torch.load(
            self.mfile, map_location=torch.device('cpu'))
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()

    def process_tensor(self, img_t, return_face=True):
        b, c, h, w = img_t.shape
        img_t = F.interpolate(img_t, (self.resolution, self.resolution))

        with torch.no_grad():
            out, __ = self.model(img_t)

        out = F.interpolate(out, (w, h))
        return out

    def process(self, ims, return_face=True):
        res = []
        faces = []
        for i in range(0, len(ims), self.batchSize):
            sizes = []
            imt = None
            for im in ims[i:i + self.batchSize]:
                sizes.append(im.shape[0])
                im = cv2.resize(im, (self.resolution, self.resolution))
                im_pil = Image.fromarray(im)
                imt = self.img2tensor(im_pil) if imt is None else torch.cat(
                    (imt, self.img2tensor(im_pil)), dim=0)

            imt = torch.flip(imt, [1])
            with torch.no_grad():
                img_outs, __ = self.model(imt)

            for sz, img_out in zip(sizes, img_outs):
                img = self.tensor2img(img_out)
                if return_face:
                    faces.append(img)
                img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_AREA)
                res.append(img)

        return res, faces

    def img2tensor(self, img):
        img_t = self.transform(img).to(self.device)
        img_t = torch.unsqueeze(img_t, 0)
        return img_t

    def tensor2img(self, image_tensor, bytes=255.0, imtype=np.uint8):
        if image_tensor.dim() == 3:
            image_numpy = image_tensor.cpu().float().numpy()
        else:
            image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        image_numpy = image_numpy[:, :, ::-1]
        image_numpy = np.clip(
            image_numpy * np.asarray([0.5, 0.5, 0.5])
            + np.asarray([0.5, 0.5, 0.5]), 0, 1)
        image_numpy = image_numpy * bytes
        return image_numpy.astype(imtype)
