# Part of the implementation is borrowed and modified from BMT and video_features,
# publicly available at https://github.com/v-iashin/BMT
# and https://github.com/v-iashin/video_features

import argparse
import os
import os.path as osp
from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from bmt_clipit.sample.single_video_prediction import (caption_proposals,
                                                       generate_proposals,
                                                       load_cap_model,
                                                       load_prop_model)
from bmt_clipit.utilities.proposal_utils import non_max_suppresion
from torch.nn.parallel import DataParallel, DistributedDataParallel
from videofeatures_clipit.models.i3d.extract_i3d import ExtractI3D
from videofeatures_clipit.models.vggish.extract_vggish import ExtractVGGish
from videofeatures_clipit.utils.utils import (fix_tensorflow_gpu_allocation,
                                              form_list_from_user_input)

from modelscope.metainfo import Models
from modelscope.models.base import Tensor, TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.language_guided_video_summarization.transformer import \
    Transformer
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


def extract_text(args):
    # Loading models and other essential stuff
    cap_cfg, cap_model, train_dataset = load_cap_model(
        args.pretrained_cap_model_path, args.device_id)
    prop_cfg, prop_model = load_prop_model(args.device_id,
                                           args.prop_generator_model_path,
                                           args.pretrained_cap_model_path,
                                           args.max_prop_per_vid)
    # Proposal
    proposals = generate_proposals(prop_model, args.features,
                                   train_dataset.pad_idx, prop_cfg,
                                   args.device_id, args.duration_in_secs)
    # NMS if specified
    if args.nms_tiou_thresh is not None:
        proposals = non_max_suppresion(proposals.squeeze(),
                                       args.nms_tiou_thresh)
        proposals = proposals.unsqueeze(0)
    # Captions for each proposal
    captions = caption_proposals(cap_model, args.features, train_dataset,
                                 cap_cfg, args.device_id, proposals,
                                 args.duration_in_secs)
    return captions


def extract_video_features(video_path, tmp_path, feature_type, i3d_flow_path,
                           i3d_rgb_path, kinetics_class_labels, pwc_path,
                           vggish_model_path, vggish_pca_path, extraction_fps,
                           device):
    default_args = dict(
        device=device,
        extraction_fps=extraction_fps,
        feature_type=feature_type,
        file_with_video_paths=None,
        i3d_flow_path=i3d_flow_path,
        i3d_rgb_path=i3d_rgb_path,
        keep_frames=False,
        kinetics_class_labels=kinetics_class_labels,
        min_side_size=256,
        pwc_path=pwc_path,
        show_kinetics_pred=False,
        stack_size=64,
        step_size=64,
        tmp_path=tmp_path,
        vggish_model_path=vggish_model_path,
        vggish_pca_path=vggish_pca_path,
    )
    args = argparse.Namespace(**default_args)

    if args.feature_type == 'i3d':
        extractor = ExtractI3D(args)
    elif args.feature_type == 'vggish':
        extractor = ExtractVGGish(args)

    feats = extractor(video_path)
    return feats


def video_features_to_txt(duration_in_secs, pretrained_cap_model_path,
                          prop_generator_model_path, features, device_id):
    default_args = dict(
        device_id=device_id,
        duration_in_secs=duration_in_secs,
        features=features,
        pretrained_cap_model_path=pretrained_cap_model_path,
        prop_generator_model_path=prop_generator_model_path,
        max_prop_per_vid=100,
        nms_tiou_thresh=0.4,
    )
    args = argparse.Namespace(**default_args)
    txt = extract_text(args)
    return txt


@MODELS.register_module(
    Tasks.language_guided_video_summarization,
    module_name=Models.language_guided_video_summarization)
class ClipItVideoSummarization(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the video summarization model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """
        super().__init__(model_dir, *args, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        self.loss = nn.MSELoss()
        self.model = Transformer()
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        self.model = self.model.to(self._device)

        self.model = self._load_pretrained(self.model, model_path)

        if self.training:
            self.model.train()
        else:
            self.model.eval()

    def _train_forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        frame_features = input['frame_features']
        txt_features = input['txt_features']
        gtscore = input['gtscore']
        preds, attn_weights = self.model(frame_features, txt_features,
                                         frame_features)
        return {'loss': self.loss(preds, gtscore)}

    def _inference_forward(self, input: Dict[str,
                                             Tensor]) -> Dict[str, Tensor]:
        frame_features = input['frame_features']
        txt_features = input['txt_features']
        y, dec_output = self.model(frame_features, txt_features,
                                   frame_features)
        return {'scores': y}

    def forward(self, input: Dict[str,
                                  Tensor]) -> Dict[str, Union[list, Tensor]]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Union[list, Tensor]]: results
        """
        for key, value in input.items():
            input[key] = input[key].to(self._device)
        if self.training:
            return self._train_forward(input)
        else:
            return self._inference_forward(input)
