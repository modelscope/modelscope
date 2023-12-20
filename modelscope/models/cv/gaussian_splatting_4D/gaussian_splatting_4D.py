import math
import os
import os.path as osp
import sys
from typing import Any, Dict

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from shotdetect_scenedetect_lgss import shot_detector
from tqdm import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks
from modelscope.utils.logger import get_logger

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import imageio
import numpy as np
import torch
import os
import cv2
from tqdm import tqdm
from os import makedirs
from .gaussian_renderer.render import render
from .scene.scene import Scene

import torchvision
from .utils.general_utils import safe_state
from argparse import ArgumentParser
from .arguments.params import ModelParams, PipelineParams, get_combined_args, ModelHiddenParams
# from .gaussian_renderer import GaussianModel
from .scene.gaussian_model import GaussianModel

from time import time

import mmcv
from .utils.params_utils import merge_hparams
logger = get_logger()


@MODELS.register_module(
    Tasks.gaussian_splatting_4D, module_name=Models.gaussian_splatting_4D)

class GaussianSplatting4D(TorchModel):
    def __init__(self, model_dir: str, *args, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, *args, **kwargs)
        self.skip_test = False
        self.skip_train = True
        self.skip_video = False
        self.configs = os.path.join(model_dir, 'bouncingballs\\bouncingballs.py')
        self.iteration = 20000




    def to8b(self, x):
        return (255 * np.clip(x.cpu().numpy(), 0, 1)).astype(np.uint8)

    def render_set(self, model_path, name, iteration, views, gaussians, pipeline, background):
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

        makedirs(render_path, exist_ok=True)
        makedirs(gts_path, exist_ok=True)
        render_images = []
        gt_list = []
        render_list = []
        
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            if idx == 0:time1 = time()
            rendering = render(view, gaussians, pipeline, background)["render"]
            # torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            render_images.append(self.to8b(rendering).transpose(1,2,0))
            # print(to8b(rendering).shape)
            render_list.append(rendering)
            if name in ["train", "test"]:
                gt = view.original_image[0:3, :, :]
                # torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                gt_list.append(gt)
        time2=time()
        print("FPS:",(len(views)-1)/(time2-time1))
        count = 0
        print("writing training images.")
        if len(gt_list) != 0:
            for image in tqdm(gt_list):
                torchvision.utils.save_image(image, os.path.join(gts_path, '{0:05d}'.format(count) + ".png"))
                count+=1
        count = 0
        print("writing rendering images.")
        if len(render_list) != 0:
            for image in tqdm(render_list):
                torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(count) + ".png"))
                count +=1
        
        imageio.mimwrite(os.path.join(model_path, name, "ours_{}".format(iteration), 'video_rgb.mp4'), render_images, fps=30, quality=8)
        
    def render_sets(self, dataset : ModelParams, hyperparam, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, skip_video: bool):
        with torch.no_grad():
            gaussians = GaussianModel(dataset.sh_degree, hyperparam)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            if not skip_train:
                self.render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

            if not skip_test:
                self.render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
            if not skip_video:
                self.render_set(dataset.model_path,"video",scene.loaded_iter,scene.getVideoCameras(),gaussians,pipeline,background)
    
    def render(self, model_dir, source_dir):
        parser = ArgumentParser(description="Testing script parameters")
        parser.model_path = os.path.join(model_dir, "output\\dnerf\\bouncingballs")
        parser.source_path = os.path.join(source_dir, "bouncingballs")

        args = get_combined_args(parser)
        config = mmcv.Config.fromfile(self.configs)
        args = merge_hparams(args, config)

        safe_state(True)
        self.model = ModelParams(parser, sentinel=True)
        self.pipeline = PipelineParams(parser)
        self.hyperparam = ModelHiddenParams(parser)
        self.render_sets(self.model.extract(args), self.hyperparam.extract(args), self.iteration, self.pipeline.extract(args), self.skip_train, self.skip_test, self.skip_video)
