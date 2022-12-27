# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import random
import shutil
import tempfile
from typing import Any, Dict

import clip
import cv2
import numpy as np
import torch
from PIL import Image

from modelscope.metainfo import Pipelines
from modelscope.models.cv.language_guided_video_summarization import \
    ClipItVideoSummarization
from modelscope.models.cv.language_guided_video_summarization.summarizer import (
    extract_video_features, video_features_to_txt)
from modelscope.models.cv.video_summarization import summary_format
from modelscope.models.cv.video_summarization.summarizer import (
    generate_summary, get_change_points)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.language_guided_video_summarization,
    module_name=Pipelines.language_guided_video_summarization)
class LanguageGuidedVideoSummarizationPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a language guided video summarization pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, auto_collate=False, **kwargs)
        logger.info(f'loading model from {model}')
        self.model_dir = self.model.model_dir

        self.tmp_dir = kwargs.get('tmp_dir', None)
        if self.tmp_dir is None:
            self.tmp_dir = tempfile.TemporaryDirectory().name

        config_path = osp.join(model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.cfg = Config.from_file(config_path)

        self.clip_model, self.clip_preprocess = clip.load(
            'ViT-B/32',
            device=self.device,
            download_root=os.path.join(self.model_dir, 'clip'))

        self.clipit_model = ClipItVideoSummarization(model)
        self.clipit_model = self.clipit_model.to(self.device).eval()

        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if not isinstance(input, tuple):
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')

        video_path, sentences = input

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        frames = []
        picks = []
        cap = cv2.VideoCapture(video_path)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_idx = 0
        # extract 1 frame every 15 frames in the video and save the frame index
        while (cap.isOpened()):
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 15 == 0:
                frames.append(frame)
                picks.append(frame_idx)
            frame_idx += 1
        n_frame = frame_idx

        if sentences is None or len(sentences) == 0:
            logger.info('input sentences is none, using sentences from video!')

            tmp_path = os.path.join(self.tmp_dir, 'tmp')
            i3d_flow_path = os.path.join(self.model_dir, 'i3d/i3d_flow.pt')
            i3d_rgb_path = os.path.join(self.model_dir, 'i3d/i3d_rgb.pt')
            kinetics_class_labels = os.path.join(self.model_dir,
                                                 'i3d/label_map.txt')
            pwc_path = os.path.join(self.model_dir, 'i3d/pwc_net.pt')
            vggish_model_path = os.path.join(self.model_dir,
                                             'vggish/vggish_model.ckpt')
            vggish_pca_path = os.path.join(self.model_dir,
                                           'vggish/vggish_pca_params.npz')

            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            i3d_feats = extract_video_features(
                video_path=video_path,
                feature_type='i3d',
                tmp_path=tmp_path,
                i3d_flow_path=i3d_flow_path,
                i3d_rgb_path=i3d_rgb_path,
                kinetics_class_labels=kinetics_class_labels,
                pwc_path=pwc_path,
                vggish_model_path=vggish_model_path,
                vggish_pca_path=vggish_pca_path,
                extraction_fps=2,
                device=device)
            rgb = i3d_feats['rgb']
            flow = i3d_feats['flow']

            device = '/gpu:0' if torch.cuda.is_available() else '/cpu:0'
            vggish = extract_video_features(
                video_path=video_path,
                feature_type='vggish',
                tmp_path=tmp_path,
                i3d_flow_path=i3d_flow_path,
                i3d_rgb_path=i3d_rgb_path,
                kinetics_class_labels=kinetics_class_labels,
                pwc_path=pwc_path,
                vggish_model_path=vggish_model_path,
                vggish_pca_path=vggish_pca_path,
                extraction_fps=2,
                device=device)
            audio = vggish['audio']

            duration_in_secs = float(self.frame_count) / self.fps

            txt = video_features_to_txt(
                duration_in_secs=duration_in_secs,
                pretrained_cap_model_path=os.path.join(
                    self.model_dir, 'bmt/sample/best_cap_model.pt'),
                prop_generator_model_path=os.path.join(
                    self.model_dir, 'bmt/sample/best_prop_model.pt'),
                features={
                    'rgb': rgb,
                    'flow': flow,
                    'audio': audio
                },
                device_id=0)
            sentences = [item['sentence'] for item in txt]

        clip_image_features = []
        for frame in frames:
            x = self.clip_preprocess(
                Image.fromarray(cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                f = self.clip_model.encode_image(x).squeeze(0).cpu().numpy()
            clip_image_features.append(f)

        clip_txt_features = []
        for sentence in sentences:
            text_input = clip.tokenize(sentence).to(self.device)
            with torch.no_grad():
                text_feature = self.clip_model.encode_text(text_input).squeeze(
                    0).cpu().numpy()
            clip_txt_features.append(text_feature)
        clip_txt_features = self.sample_txt_feateures(clip_txt_features)
        clip_txt_features = np.array(clip_txt_features).reshape((1, -1))

        result = {
            'video_name': video_path,
            'clip_image_features': np.array(clip_image_features),
            'clip_txt_features': np.array(clip_txt_features),
            'n_frame': n_frame,
            'picks': np.array(picks)
        }
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        clip_image_features = input['clip_image_features']
        clip_txt_features = input['clip_txt_features']
        clip_image_features = self.norm_feature(clip_image_features)
        clip_txt_features = self.norm_feature(clip_txt_features)

        change_points, n_frame_per_seg = get_change_points(
            clip_image_features, input['n_frame'])

        summary = self.inference(clip_image_features, clip_txt_features,
                                 input['n_frame'], input['picks'],
                                 change_points)

        output = summary_format(summary, self.fps)

        return {OutputKeys.OUTPUT: output}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if os.path.exists(self.tmp_dir):
            shutil.rmtree(self.tmp_dir)
        return inputs

    def inference(self, clip_image_features, clip_txt_features, n_frames,
                  picks, change_points):
        clip_image_features = torch.from_numpy(
            np.array(clip_image_features, np.float32)).unsqueeze(0)
        clip_txt_features = torch.from_numpy(
            np.array(clip_txt_features, np.float32)).unsqueeze(0)
        picks = np.array(picks, np.int32)

        with torch.no_grad():
            results = self.clipit_model(
                dict(
                    frame_features=clip_image_features,
                    txt_features=clip_txt_features))
            scores = results['scores']
            if not scores.device.type == 'cpu':
                scores = scores.cpu()
            scores = scores.squeeze(0).numpy().tolist()
            summary = generate_summary([change_points], [scores], [n_frames],
                                       [picks])[0]

        return summary.tolist()

    def sample_txt_feateures(self, feat, num=7):
        while len(feat) < num:
            feat.append(feat[-1])
        idxes = list(np.arange(0, len(feat)))
        samples_idx = []
        for ii in range(num):
            idx = random.choice(idxes)
            while idx in samples_idx:
                idx = random.choice(idxes)
            samples_idx.append(idx)
        samples_idx.sort()

        samples = []
        for idx in samples_idx:
            samples.append(feat[idx])
        return samples

    def norm_feature(self, frames_feat):
        for ii in range(len(frames_feat)):
            frame_feat = frames_feat[ii]
            frames_feat[ii] = frame_feat / np.linalg.norm(frame_feat)
        frames_feat = frames_feat.reshape((frames_feat.shape[0], -1))
        return frames_feat
