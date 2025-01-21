# The implementation is based on FairFace, available at
# https://github.com/dchen236/FairFace
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, models, transforms

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.face_attribute_recognition, module_name=Models.fairface)
class FaceAttributeRecognition(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device
        self.cfg_path = model_path.replace(ModelFile.TORCH_MODEL_FILE,
                                           ModelFile.CONFIGURATION)
        fair_face = torchvision.models.resnet34(pretrained=False)
        fair_face.fc = nn.Linear(fair_face.fc.in_features, 18)
        self.net = fair_face
        self.load_model()
        self.net = self.net.to(device)
        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        self.net.load_state_dict(pretrained_dict, strict=True)
        self.net.eval()

    def forward(self, img):
        """ FariFace model forward process.

        Args:
            img: [h, w, c]

        Return:
            list of attribute result: [gender_score, age_score]
        """
        img = cv2.cvtColor(img.cpu().numpy(), cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)

        inputs = self.trans(img)

        c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to(self.device)
        inputs = Variable(inputs, volatile=True)
        outputs = self.net(inputs)[0]

        gender_outputs = outputs[7:9]
        age_outputs = outputs[9:18]

        gender_score = F.softmax(gender_outputs).detach().cpu().tolist()
        age_score = F.softmax(age_outputs).detach().cpu().tolist()

        return [gender_score, age_score]
