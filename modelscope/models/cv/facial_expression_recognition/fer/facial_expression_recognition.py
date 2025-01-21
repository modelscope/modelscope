# The implementation is based on Facial-Expression-Recognition, available at
# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks
from . import transforms
from .vgg import VGG


@MODELS.register_module(
    Tasks.facial_expression_recognition, module_name=Models.fer)
class FacialExpressionRecognition(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.cfg_path = model_path.replace(ModelFile.TORCH_MODEL_FILE,
                                           ModelFile.CONFIGURATION)
        self.net = VGG('VGG19', cfg_path=self.cfg_path)
        self.load_model()
        self.net = self.net.to(device)
        self.transform_test = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.mean = np.array([[104, 117, 123]])

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict['net'], strict=True)
        self.net.eval()

    def forward(self, input):
        img = input['img']
        img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(np.uint8(img))
        inputs = self.transform_test(img)

        ncrops, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to(self.device)
        inputs = Variable(inputs, volatile=True)
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg)
        _, predicted = torch.max(outputs_avg.data, 0)

        return score, predicted
