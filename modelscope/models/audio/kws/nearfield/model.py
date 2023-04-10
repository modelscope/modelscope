# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
import tempfile
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.audio.audio_utils import update_conf
from modelscope.utils.constant import Tasks
from .cmvn import GlobalCMVN, load_kaldi_cmvn
from .fsmn import FSMN


@MODELS.register_module(
    Tasks.keyword_spotting,
    module_name=Models.speech_kws_fsmn_char_ctc_nearfield)
class FSMNDecorator(TorchModel):
    r""" A decorator of FSMN for integrating into modelscope framework """

    def __init__(self,
                 model_dir: str,
                 cmvn_file: str = None,
                 backbone: dict = None,
                 input_dim: int = 400,
                 output_dim: int = 2599,
                 training: Optional[bool] = False,
                 *args,
                 **kwargs):
        """initialize the fsmn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            cmvn_file (str): cmvn file
            backbone (dict): params related to backbone
            input_dim (int): input dimension of network
            output_dim (int): output dimension of network
            training (bool): training or inference mode
        """
        super().__init__(model_dir, *args, **kwargs)

        self.model = None
        self.model_cfg = None

        if training:
            self.model = self.init_model(cmvn_file, backbone, input_dim,
                                         output_dim)
        else:
            self.model_cfg = {
                'model_workspace': model_dir,
                'config_path': os.path.join(model_dir, 'config.yaml')
            }

    def __del__(self):
        if hasattr(self, 'tmp_dir'):
            self.tmp_dir.cleanup()

    def forward(self, input) -> Dict[str, Tensor]:
        """
        Args:
            input (torch.Tensor): Input tensor (B, T, D)
        """
        if self.model is not None and input is not None:
            return self.model.forward(input)
        else:
            return self.model_cfg

    def init_model(self, cmvn_file, backbone, input_dim, output_dim):
        if cmvn_file is not None:
            mean, istd = load_kaldi_cmvn(cmvn_file)
            global_cmvn = GlobalCMVN(
                torch.from_numpy(mean).float(),
                torch.from_numpy(istd).float(),
            )
        else:
            global_cmvn = None

        hidden_dim = 128
        preprocessing = None

        input_affine_dim = backbone['input_affine_dim']
        num_layers = backbone['num_layers']
        linear_dim = backbone['linear_dim']
        proj_dim = backbone['proj_dim']
        left_order = backbone['left_order']
        right_order = backbone['right_order']
        left_stride = backbone['left_stride']
        right_stride = backbone['right_stride']
        output_affine_dim = backbone['output_affine_dim']
        backbone = FSMN(input_dim, input_affine_dim, num_layers, linear_dim,
                        proj_dim, left_order, right_order, left_stride,
                        right_stride, output_affine_dim, output_dim)

        classifier = None
        activation = None

        kws_model = KWSModel(input_dim, output_dim, hidden_dim, global_cmvn,
                             preprocessing, backbone, classifier, activation)
        return kws_model


class KWSModel(nn.Module):
    """Our model consists of four parts:
    1. global_cmvn: Optional, (idim, idim)
    2. preprocessing: feature dimension projection, (idim, hdim)
    3. backbone: backbone or feature extractor of the whole network, (hdim, hdim)
    4. classifier: output layer or classifier of KWS model, (hdim, odim)
    5. activation:
        nn.Sigmoid for wakeup word
        nn.Identity for speech command dataset
    """

    def __init__(
        self,
        idim: int,
        odim: int,
        hdim: int,
        global_cmvn: Optional[nn.Module],
        preprocessing: Optional[nn.Module],
        backbone: nn.Module,
        classifier: nn.Module,
        activation: nn.Module,
    ):
        """
        Args:
            idim (int): input dimension of network
            odim (int): output dimension of network
            hdim (int): hidden dimension of network
            global_cmvn (nn.Module): cmvn for input feature, (idim, idim)
            preprocessing (nn.Module): feature dimension projection, (idim, hdim)
            backbone (nn.Module): backbone or feature extractor of the whole network, (hdim, hdim)
            classifier (nn.Module): output layer or classifier of KWS model, (hdim, odim)
            activation (nn.Module): nn.Identity for training, nn.Sigmoid for inference
        """
        super().__init__()
        self.idim = idim
        self.odim = odim
        self.hdim = hdim
        self.global_cmvn = global_cmvn
        self.preprocessing = preprocessing
        self.backbone = backbone
        self.classifier = classifier
        self.activation = activation

    def to_kaldi_net(self):
        return self.backbone.to_kaldi_net()

    def to_pytorch_net(self, kaldi_file):
        return self.backbone.to_pytorch_net(kaldi_file)

    def forward(
        self,
        x: torch.Tensor,
        in_cache: torch.Tensor = torch.zeros(0, 0, 0, dtype=torch.float)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.global_cmvn is not None:
            x = self.global_cmvn(x)
        if self.preprocessing is not None:
            x = self.preprocessing(x)

        x, out_cache = self.backbone(x, in_cache)

        if self.classifier is not None:
            x = self.classifier(x)
        if self.activation is not None:
            x = self.activation(x)
        return x, out_cache

    def fuse_modules(self):
        if self.preprocessing is not None:
            self.preprocessing.fuse_modules()
        self.backbone.fuse_modules()
