# Copyright (c) Alibaba, Inc. and its affiliates.

import os

import torch

from modelscope.metainfo import Models
from modelscope.models import MODELS, TorchModel
from modelscope.utils.constant import ModelFile, Tasks
from .sepreformer_layers import TSepReformerSeparator


@MODELS.register_module(
    Tasks.speech_separation,
    module_name=Models.
    speech_flatsepreformer_separation_temporal_8k_base_libri2mix100)
class TSepReformer(TorchModel):
    """T-SepReformer speech separation model.

    arxiv: https://arxiv.org/abs/2508.19528
    INTERSPEECH 2025: https://www.isca-archive.org/interspeech_2025/wang25j_interspeech.html

    The separator is fully time-domain: it embeds its own waveform encoder and
    decoder, so no spectral front end is needed.

    Args:
        model_dir (str): the model path.
        separator_conf (dict): arguments for :class:`TSepReformerSeparator`.
        normalize_variance (bool): divide the mixture by its standard deviation
            before the separator and restore the scale afterwards.
    """

    def __init__(self,
                 model_dir: str,
                 separator_conf: dict,
                 normalize_variance: bool = False,
                 *args,
                 **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.normalize_variance = normalize_variance
        self.separator = TSepReformerSeparator(**separator_conf)
        self.num_spks = self.separator.num_spk

    def forward(self, input):
        """Separate a mixture waveform.

        Args:
            input (torch.Tensor): mixture waveform, [B, L].

        Returns:
            torch.Tensor: separated waveforms, [B, L, num_spks].
        """
        if self.normalize_variance:
            mix_std = torch.std(input, dim=1, keepdim=True)
            input = input / mix_std

        waves = self.separator(input)

        if self.normalize_variance:
            waves = [w * mix_std for w in waves]

        return torch.stack(waves, dim=-1)

    def load_check_point(self, load_path=None, device=None):
        if not load_path:
            load_path = self.model_dir
        if not device:
            device = torch.device('cpu')
        state_dict = torch.load(
            os.path.join(load_path, ModelFile.TORCH_MODEL_FILE),
            map_location=device,
            weights_only=True)
        self.load_state_dict(state_dict, strict=True)
