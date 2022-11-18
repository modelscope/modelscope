# The implementation is based on mtcnn, available at https://github.com/TropComplique/mtcnn-pytorch
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks
from .box_utils import calibrate_box, convert_to_square, get_image_boxes, nms
from .first_stage import run_first_stage
from .get_nets import ONet, PNet, RNet


@MODELS.register_module(Tasks.face_detection, module_name=Models.mtcnn)
class MtcnnFaceDetector(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.device = device

        self.pnet = PNet(model_path=os.path.join(self.model_path, 'pnet.npy'))
        self.rnet = RNet(model_path=os.path.join(self.model_path, 'rnet.npy'))
        self.onet = ONet(model_path=os.path.join(self.model_path, 'onet.npy'))

        self.pnet = self.pnet.to(device)
        self.rnet = self.rnet.to(device)
        self.onet = self.onet.to(device)

    def forward(self, input):
        image = Image.fromarray(np.uint8(input['img'].cpu().numpy()))
        pnet = self.pnet
        rnet = self.rnet
        onet = self.onet
        onet.eval()

        min_face_size = 20.0
        thresholds = [0.7, 0.8, 0.9]
        nms_thresholds = [0.7, 0.7, 0.7]

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes = run_first_stage(
                image,
                pnet,
                scale=s,
                threshold=thresholds[0],
                device=self.device)
            bounding_boxes.append(boxes)

        # collect boxes (and offsets, and scores) from different scales
        bounding_boxes = [i for i in bounding_boxes if i is not None]
        bounding_boxes = np.vstack(bounding_boxes)

        keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
        bounding_boxes = bounding_boxes[keep]

        # use offsets predicted by pnet to transform bounding boxes
        bounding_boxes = calibrate_box(bounding_boxes[:, 0:5],
                                       bounding_boxes[:, 5:])
        # shape [n_boxes, 5]

        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 2

        img_boxes = get_image_boxes(bounding_boxes, image, size=24)
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = rnet(img_boxes.to(self.device))
        offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[1])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
        offsets = offsets[keep]

        keep = nms(bounding_boxes, nms_thresholds[1])
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
        bounding_boxes = convert_to_square(bounding_boxes)
        bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

        # STAGE 3

        img_boxes = get_image_boxes(bounding_boxes, image, size=48)
        if len(img_boxes) == 0:
            return [], []
        img_boxes = Variable(torch.FloatTensor(img_boxes), volatile=True)
        output = onet(img_boxes.to(self.device))
        landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
        offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
        probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

        keep = np.where(probs[:, 1] > thresholds[2])[0]
        bounding_boxes = bounding_boxes[keep]
        bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
        offsets = offsets[keep]
        landmarks = landmarks[keep]

        # compute landmark points
        width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
        height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
        xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
        landmarks[:, 0:5] = np.expand_dims(
            xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
        landmarks[:, 5:10] = np.expand_dims(
            ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

        bounding_boxes = calibrate_box(bounding_boxes, offsets)
        keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
        bounding_boxes = bounding_boxes[keep]
        landmarks = landmarks[keep]
        landmarks = landmarks.reshape(-1, 2, 5).transpose(
            (0, 2, 1)).reshape(-1, 10)

        return bounding_boxes, landmarks
