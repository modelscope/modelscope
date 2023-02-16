# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
import tempfile
from typing import Any, Dict

import numpy as np
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.motion_generation import (ClassifierFreeSampleModel,
                                                    create_model,
                                                    load_model_wo_clip)
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.cv.motion_utils.motion_process import recover_from_ric
from modelscope.utils.cv.motion_utils.plot_script import plot_3d_motion
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.motion_generation, module_name=Pipelines.motion_generattion)
class MDMMotionGeneration(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create motion generation pipeline for prediction
        Args:
            model: model id on modelscope hub.
        """
        super().__init__(model=model, **kwargs)
        model_path = osp.join(self.model, ModelFile.TORCH_MODEL_FILE)
        logger.info(f'loading model from {model_path}')
        config_path = osp.join(self.model, ModelFile.CONFIGURATION)
        logger.info(f'loading config from {config_path}')
        self.mean = np.load(osp.join(self.model, 'Mean.npy'))
        self.std = np.load(osp.join(self.model, 'Std.npy'))
        self.cfg = Config.from_file(config_path)
        self.cfg.update({'smpl_data_path': osp.join(self.model, 'smpl')})
        self.cfg.update(kwargs)
        self.n_joints = 22
        self.fps = 20
        self.n_frames = 120
        self.mdm, self.diffusion = create_model(self.cfg)
        state_dict = torch.load(model_path, map_location='cpu')
        load_model_wo_clip(self.mdm, state_dict)
        self.mdm = ClassifierFreeSampleModel(self.mdm)
        self.mdm.to(self.device)
        self.mdm.eval()
        logger.info('load model done')

    def preprocess(self, input: Input) -> Dict[str, Any]:
        if isinstance(input, str):
            input_text = input
        else:
            raise TypeError(f'input should be a str,'
                            f'  but got {type(input)}')
        result = {'input_text': input_text}
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        texts = [input['input_text']]
        model_kwargs = {
            'y': {
                'mask': torch.ones(1, 1, 1, self.n_frames) > 0,
                'lengths': torch.tensor([self.n_frames]),
                'tokens': None,
                'text': texts,
                'scale': torch.ones(1, device=self.device) * 2.5
            }
        }
        sample_fn = self.diffusion.p_sample_loop
        sample = sample_fn(
            self.mdm,
            (1, self.mdm.njoints, self.mdm.nfeats, self.n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        sample = (sample.cpu().permute(0, 2, 3, 1) * self.std
                  + self.mean).float()
        sample = recover_from_ric(sample, self.n_joints)
        sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)

        sample = self.mdm.rot2xyz(
            x=sample,
            mask=None,
            pose_rep='xyz',
            glob=True,
            translation=True,
            jointstype='smpl',
            vertstrans=True,
            betas=None,
            beta=0,
            glob_rot=None,
            get_rotations_back=False)
        motion = sample.cpu().numpy()
        motion = motion[0].transpose(2, 0, 1)
        out = {OutputKeys.KEYPOINTS: motion, 'text': input['input_text']}
        return out

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        output_video_path = kwargs.get(
            'output_video',
            tempfile.NamedTemporaryFile(suffix='.mp4').name)
        kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
                           [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
                           [9, 13, 16, 18, 20]]
        if output_video_path is not None:
            plot_3d_motion(
                output_video_path,
                kinematic_chain,
                inputs[OutputKeys.KEYPOINTS],
                inputs.pop('text'),
                dataset='humanml',
                fps=20)
        inputs.update({OutputKeys.OUTPUT_VIDEO: output_video_path})
        return inputs
