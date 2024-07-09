# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import numpy as np
import torch
import torchvision.transforms as T

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import Tasks
from modelscope.models.cv.human_normal_estimation.networks import nnet, config


@MODELS.register_module(
    Tasks.human_normal_estimation, module_name=Models.human_normal_estimation)
class HumanNormalEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        super().__init__(model_dir, **kwargs)
        config_file = os.path.join(model_dir, 'config.txt')
        args = config.get_args(txt_file=config_file)
        args.encoder_path = os.path.join(model_dir, args.encoder_path)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.nnet = nnet.NormalNet(args=args).to(self.device)
        self.nnet_path = os.path.join(model_dir, 'ckpt/best_nnet.pt')
        if os.path.exists(self.nnet_path):
            ckpt = torch.load(self.nnet_path, map_location=self.device)['model']
            load_dict = {}
            for k, v in ckpt.items():
                if k.startswith('module.'):
                    k_ = k.replace('module.', '')
                    load_dict[k_] = v
                else:
                    load_dict[k] = v
            self.nnet.load_state_dict(load_dict)
        self.nnet.eval()

        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, inputs):
        img = inputs['img'].astype(np.float32) / 255.0
        msk = inputs['msk'].astype(np.float32) / 255.0
        bbox = inputs['bbox']

        img_h, img_w = img.shape[0:2]
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img = self.normalize(img)

        fx = fy = (max(img_h, img_h) / 2.0) / np.tan(np.deg2rad(60.0 / 2.0))
        cx = (img_h / 2.0) - 0.5
        cy = (img_w / 2.0) - 0.5

        intrins = torch.tensor([
            [fx, 0, cx + 0.5],
            [0, fy, cy + 0.5],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device).unsqueeze(0)

        pred_norm = self.nnet(img, intrins=intrins)[-1]
        pred_norm = pred_norm.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_norm = pred_norm[0, ...]
        pred_norm = pred_norm * msk[..., None]
        pred_norm = pred_norm[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        results = pred_norm
        return results

    def postprocess(self, inputs):
        normal_result = inputs
        results = {OutputKeys.NORMALS: normal_result}
        return results

    def inference(self, data):
        results = self.forward(data)
        return results
