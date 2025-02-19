# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Dict

import soundfile as sf
import torch

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.audio.vc.src.encoder import Encoder
from modelscope.models.audio.vc.src.sv_models.DTDNN import \
    SpeakerVerificationCamplus
from modelscope.models.audio.vc.src.vocoder import (ConditionGenerator,
                                                    HiFiGANGenerator)
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.utils.constant import Tasks


@MODELS.register_module(Tasks.voice_conversion, module_name=Models.unetvc_16k)
class UnetVC(TorchModel):
    r"""A decorator of FRCRN for integrating into modelscope framework"""

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the frcrn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)
        device = kwargs.get('device', 'cpu')
        self.device = device
        static_path = os.path.join(model_dir, 'static')
        self.encoder = Encoder(
            os.path.join(static_path, 'encoder_am.mvn'),
            os.path.join(static_path, 'encoder.onnx'))
        self.spk_emb = SpeakerVerificationCamplus(
            os.path.join(static_path, 'campplus_cn_common.bin'), device)
        self.converter = ConditionGenerator(
            unet=True, extra_info=True).to(device)
        G_path = os.path.join(static_path, 'converter.pth')
        self.converter.load_state_dict(
            torch.load(G_path, map_location=lambda storage, loc: storage))
        self.converter.eval()
        self.vocoder = HiFiGANGenerator().to(device)
        self.vocoder.load_state_dict(
            torch.load(
                os.path.join(static_path, 'vocoder.pth'),
                map_location=self.device)['state_dict'])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()

    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        target_wav_path = inputs['target_wav']
        source_wav_path = inputs['source_wav']
        save_wav_path = inputs['save_path']

        with torch.no_grad():
            source_enc = self.encoder.inference(source_wav_path).to(
                self.device)

            spk_emb = self.spk_emb.forward(target_wav_path).to(self.device)

            style_mc = self.encoder.get_feats(target_wav_path).to(self.device)

            coded_sp_converted_norm = self.converter(source_enc, spk_emb,
                                                     style_mc)

            wav = self.vocoder(coded_sp_converted_norm.permute([0, 2, 1]))
            if os.path.exists(save_wav_path):
                sf.write(save_wav_path,
                         wav.flatten().cpu().data.numpy(), 16000)

        return wav.flatten().cpu().data.numpy()
