# The implementation is based on resnet, available at https://github.com/biubug6/Pytorch_Retinaface
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from .models.retinaface import RetinaFace
from .utils import PriorBox, decode, decode_landm, py_cpu_nms


@MODELS.register_module(Tasks.face_detection, module_name=Models.retinaface)
class RetinaFaceDetection(TorchModel):

    def __init__(self, model_path, device='cuda'):
        super().__init__(model_path)
        cudnn.benchmark = True
        self.model_path = model_path
        self.cfg = Config.from_file(
            model_path.replace(ModelFile.TORCH_MODEL_FILE,
                               ModelFile.CONFIGURATION))['models']
        self.net = RetinaFace(cfg=self.cfg)
        self.load_model()
        self.device = device
        self.net = self.net.to(self.device)

        self.mean = torch.tensor([[[[104]], [[117]], [[123]]]]).to(device)

    def check_keys(self, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(self.net.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(
            used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def remove_prefix(self, state_dict, prefix):
        new_state_dict = dict()
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix):]] = v
            else:
                new_state_dict[k] = v
        return new_state_dict

    def load_model(self, load_to_cpu=False):
        pretrained_dict = torch.load(
            self.model_path, map_location=torch.device('cpu'))
        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'],
                                                 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
        self.check_keys(pretrained_dict)
        self.net.load_state_dict(pretrained_dict, strict=False)
        self.net.eval()

    def forward(self, input):
        img_raw = input['img'].cpu().numpy()
        img = np.float32(img_raw)

        im_height, im_width = img.shape[:2]
        ss = 1.0
        # tricky
        if max(im_height, im_width) > 1500:
            ss = 1000.0 / max(im_height, im_width)
            img = cv2.resize(img, (0, 0), fx=ss, fy=ss)
            im_height, im_width = img.shape[:2]

        scale = torch.Tensor(
            [img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        loc, conf, landms = self.net(img)  # forward pass
        del img

        confidence_threshold = 0.9
        nms_threshold = 0.4
        top_k = 5000
        keep_top_k = 750

        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(
            landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([
            im_width, im_height, im_width, im_height, im_width, im_height,
            im_width, im_height, im_width, im_height
        ])
        scale1 = scale1.to(self.device)
        landms = landms * scale1
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(
            np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]

        landms = landms.reshape((-1, 5, 2))
        landms = landms.reshape(
            -1,
            10,
        )
        return dets / ss, landms / ss
