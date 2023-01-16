# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
import tqdm

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.models.cv.video_depth_estimation.models.model_wrapper import \
    ModelWrapper
from modelscope.models.cv.video_depth_estimation.utils.augmentations import (
    resize_image, to_tensor)
from modelscope.models.cv.video_depth_estimation.utils.config import \
    parse_test_file
from modelscope.models.cv.video_depth_estimation.utils.depth import (
    inv2depth, viz_inv_depth, write_depth)
from modelscope.models.cv.video_depth_estimation.utils.image import (
    get_intrinsics, load_image, parse_video)
from modelscope.outputs import OutputKeys
from modelscope.utils.constant import ModelFile, Tasks


@MODELS.register_module(
    Tasks.video_depth_estimation,
    module_name=Models.dro_resnet18_depth_estimation)
class DROEstimation(TorchModel):

    def __init__(self, model_dir: str, **kwargs):
        """str -- model file root."""
        super().__init__(model_dir, **kwargs)

        model_path = osp.join(model_dir, ModelFile.TORCH_MODEL_FILE)

        # Parse arguments
        config, state_dict = parse_test_file(model_path)

        # If no image shape is provided, use the checkpoint one
        self.image_shape = config.datasets.augmentation.image_shape
        print(f'== input image shape:{self.image_shape}')

        # Initialize model wrapper from checkpoint arguments
        self.model_wrapper = ModelWrapper(config, load_datasets=False)
        # Restore monodepth_model state
        self.model_wrapper.load_state_dict(state_dict)

        # Send model to GPU if available
        if torch.cuda.is_available():
            model_wrapper = self.model_wrapper.to('cuda')
        else:
            model_wrapper = self.model_wrapper
            print('cuda is not available, use cpu')

        # Set to eval mode
        model_wrapper.eval()

    def forward(self, Inputs):
        return self.model_wrapper(Inputs)

    def postprocess(self, Inputs):
        return Inputs

    def inference(self, data):

        print('processing video input:.........')
        input_type = 'video'
        sample_rate = 1
        data_type = 'indoor'

        assert osp.splitext(data['video_path'])[1] in [
            '.mp4', '.avi', '.mov', '.mpeg', '.flv', '.wmv'
        ]
        input_video_images = os.path.join('tmp/input_video_images')
        parse_video(data['video_path'], input_video_images, sample_rate)
        # update input
        input = input_video_images

        files = []
        for ext in ['png', 'jpg', 'bmp']:
            files.extend(glob((os.path.join(input, '*.{}'.format(ext)))))

        if input_type == 'folder':
            print('processing folder input:...........')
            print(f'folder total frames num: {len(files)}')
            files = files[::sample_rate]

        files.sort()
        print('Found total {} files'.format(len(files)))
        assert len(files) > 2

        # Process each file
        list_of_files = list(zip(files[:-2], files[1:-1], files[2:]))

        depth_list = []
        pose_list = []
        vis_depth_list = []
        depth_upsample_list = []
        vis_depth_upsample_list = []

        print(f'*********************data_type:{data_type}')
        print('inference start.....................')
        for fn1, fn2, fn3 in tqdm.tqdm(list_of_files):
            depth, vis_depth, depth_upsample, vis_depth_upsample, pose21, pose23, intr, rgb = self.infer_and_save_pose(
                [fn1, fn3], fn2, self.model_wrapper, self.image_shape,
                data_type)
            pose_list.append((pose21, pose23))
            depth_list.append(depth)
            vis_depth_list.append(vis_depth.astype(np.uint8))
            depth_upsample_list.append(depth_upsample)
            vis_depth_upsample_list.append(vis_depth_upsample.astype(np.uint8))

        return {
            'depths': depth_list,
            'depths_color': vis_depth_upsample_list,
            'poses': pose_list
        }

    @torch.no_grad()
    def infer_and_save_pose(self, input_file_refs, input_file, model_wrapper,
                            image_shape, data_type):
        """
        Process a single input file to produce and save visualization

        Parameters
        ----------
        input_file_refs : list(str)
            Reference image file paths
        input_file : str
            Image file for pose estimation
        model_wrapper : nn.Module
            Model wrapper used for inference
        image_shape : Image shape
            Input image shape
        half: bool
            use half precision (fp16)
        save: str
            Save format (npz or png)
        """

        image_raw_wh = load_image(input_file).size

        # Load image
        def process_image(filename):
            image = load_image(filename)
            # Resize and to tensor
            intr = get_intrinsics(image.size, image_shape, data_type)  # (3, 3)
            image = resize_image(image, image_shape)
            image = to_tensor(image).unsqueeze(0)
            intr = torch.from_numpy(intr).unsqueeze(0)  # (1, 3, 3)
            # Send image to GPU if available
            if torch.cuda.is_available():
                image = image.to('cuda')
                intr = intr.to('cuda')
            return image, intr

        image_ref = [
            process_image(input_file_ref)[0]
            for input_file_ref in input_file_refs
        ]
        image, intrinsics = process_image(input_file)

        batch = {
            'rgb': image,
            'rgb_context': image_ref,
            'intrinsics': intrinsics
        }

        output = self.forward(batch)
        inv_depth = output['inv_depths'][0]  # (1, 1, h, w)
        depth = inv2depth(inv_depth)[0, 0].detach().cpu().numpy()  # (h, w)

        pose21 = output['poses'][0].mat[0].detach().cpu().numpy(
        )  # (4, 4)  # TODO check: targe -> ref[0]
        pose23 = output['poses'][1].mat[0].detach().cpu().numpy(
        )  # (4, 4)  # TODO check: targe -> ref[0]

        vis_depth = viz_inv_depth(inv_depth[0]) * 255

        vis_depth_upsample = cv2.resize(
            vis_depth, image_raw_wh, interpolation=cv2.INTER_LINEAR)
        depth_upsample = cv2.resize(
            depth, image_raw_wh, interpolation=cv2.INTER_NEAREST)

        return depth, vis_depth, depth_upsample, vis_depth_upsample, pose21, pose23, intrinsics[
            0].detach().cpu().numpy(), image[0].permute(
                1, 2, 0).detach().cpu().numpy() * 255
