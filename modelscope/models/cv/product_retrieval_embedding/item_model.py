# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.

import os.path as osp
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.product_retrieval_embedding.item_detection import \
    YOLOXONNX
from modelscope.models.cv.product_retrieval_embedding.item_embedding import (
    preprocess, resnet50_embed)
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger

logger = get_logger()

__all__ = ['ProductRetrievalEmbedding']


@MODELS.register_module(
    Tasks.product_retrieval_embedding,
    module_name=Models.product_retrieval_embedding)
class ProductRetrievalEmbedding(TorchModel):

    def __init__(self, model_dir, device='cpu', **kwargs):
        super().__init__(model_dir=model_dir, device=device, **kwargs)

        def filter_param(src_params, own_state):
            copied_keys = []
            for name, param in src_params.items():
                if 'module.' == name[0:7]:
                    name = name[7:]
                if '.module.' not in list(own_state.keys())[0]:
                    name = name.replace('.module.', '.')
                if (name in own_state) and (own_state[name].shape
                                            == param.shape):
                    own_state[name].copy_(param)
                    copied_keys.append(name)

        def load_pretrained(model, src_params):
            if 'state_dict' in src_params:
                src_params = src_params['state_dict']
            own_state = model.state_dict()
            filter_param(src_params, own_state)
            model.load_state_dict(own_state)

        self.device = create_device(
            device)  # device.type == "cpu" or device.type == "cuda"
        self.use_gpu = self.device.type == 'cuda'

        # config the model path
        self.local_model_dir = model_dir

        # init feat model
        self.preprocess_for_embed = preprocess  # input is cv2 bgr format
        model_feat = resnet50_embed()
        src_params = torch.load(
            osp.join(self.local_model_dir, ModelFile.TORCH_MODEL_BIN_FILE),
            'cpu')
        load_pretrained(model_feat, src_params)
        if self.use_gpu:
            model_feat.to(self.device)
            logger.info('Use GPU: {}'.format(self.device))
        else:
            logger.info('Use CPU for inference')

        self.model_feat = model_feat

        # init det model
        self.model_det = YOLOXONNX(
            onnx_path=osp.join(self.local_model_dir, 'onnx_detection.onnx'),
            multi_detect=False)
        logger.info('load model done')

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """
        detection and feature extraction for input product image
        """
        # input should be cv2 bgr format
        assert 'img' in input.keys()

        def set_phase(model, is_train):
            if is_train:
                model.train()
            else:
                model.eval()

        is_train = False
        set_phase(self.model_feat, is_train)
        img = input['img']  # for detection
        cid = '3'  # preprocess detection category bag
        # transform img(tensor) to numpy array with bgr
        if isinstance(img, torch.Tensor):
            img = img.data.cpu().numpy()
        res, crop_img = self.model_det.forward(img,
                                               cid)  # detect with bag category
        crop_img = self.preprocess_for_embed(crop_img)  # feat preprocess
        input_tensor = torch.from_numpy(crop_img.astype(np.float32))
        device = next(self.model_feat.parameters()).device
        use_gpu = device.type == 'cuda'
        with torch.no_grad():
            if use_gpu:
                input_tensor = input_tensor.to(device)
            out_embedding = self.model_feat(input_tensor)
            out_embedding = out_embedding.cpu().numpy()[
                0, :]  # feature array with 512 elements

        output = {OutputKeys.IMG_EMBEDDING: None}
        output[OutputKeys.IMG_EMBEDDING] = out_embedding
        return output
