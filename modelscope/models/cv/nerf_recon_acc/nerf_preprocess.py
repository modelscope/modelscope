# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import subprocess
from typing import Any, Dict, Union

import cv2
import numpy as np
import tensorflow as tf

from modelscope.metainfo import Preprocessors
from modelscope.preprocessors import Preprocessor
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.utils.constant import Fields, ModeKeys
from modelscope.utils.logger import get_logger

logger = get_logger()


@PREPROCESSORS.register_module(
    Fields.cv, module_name=Preprocessors.nerf_recon_acc_preprocessor)
class NeRFReconPreprocessor(Preprocessor):

    def __init__(self,
                 mode=ModeKeys.INFERENCE,
                 data_type='colmap',
                 use_mask=True,
                 match_type='exhaustive_matcher',
                 frame_count=60,
                 use_distortion=False,
                 *args,
                 **kwargs):

        super().__init__(mode)

        # set preprocessor info
        self.data_type = data_type
        self.use_mask = use_mask

        self.match_type = match_type
        if match_type != 'exhaustive_matcher' and match_type != 'sequential_matcher':
            raise Exception('matcher type {} is not valid'.format(match_type))
        self.frame_count = frame_count
        self.use_distortion = use_distortion

    def __call__(self, data: Union[str, Dict], **kwargs) -> Dict[str, Any]:

        if self.data_type != 'blender' and self.data_type != 'colmap':
            raise Exception('data type {} is not support currently'.format(
                self.data_type))

        data_dir = data['data_dir']
        os.makedirs(data_dir, exist_ok=True)
        if self.data_type == 'blender':
            transform_file = os.path.join(data_dir, 'transforms_train.json')
            if not os.path.exists(transform_file):
                raise Exception('Blender dataset is not found')

        if self.data_type == 'colmap':
            video_path = data['video_input_path']
            if video_path != '':
                self.split_frames(video_path, data_dir, self.frame_count)
            self.gen_poses(data_dir, self.match_type, self.use_distortion)
            files_needed = [
                '{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']
            ]
            if self.use_distortion:
                colmap_dir = os.path.join(data_dir, 'preprocess/sparse')
                files_had = os.listdir(colmap_dir)
            else:
                colmap_dir = os.path.join(data_dir, 'sparse/0')
                files_had = os.listdir(colmap_dir)
            if not all([f in files_had for f in files_needed]):
                raise Exception('colmap run failed')

        data = {}
        data['data_dir'] = data_dir
        return data

    def split_frames(self, video_path, basedir, frame_count=60):
        cap = cv2.VideoCapture(video_path)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        frame_total = round(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not os.path.exists(os.path.join(basedir, 'images')):
            logger.info('Need to run ffmpeg')
            image_dir = os.path.join(basedir, 'images')
            os.makedirs(image_dir, exist_ok=True)
            fps = int(frame_count * fps / frame_total)
            cmd = f"ffmpeg -i {video_path} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {image_dir}/%04d.png"
            os.system(cmd)
            logger.info('split frames done')
        else:
            logger.info('Don\'t need to run ffmpeg')

    def run_colmap(self, basedir, match_type, use_distortion):
        logfile_name = os.path.join(basedir, 'colmap_output.txt')
        logfile = open(logfile_name, 'w')

        feature_extractor_args = [
            'colmap', 'feature_extractor', '--database_path',
            os.path.join(basedir, 'database.db'), '--image_path',
            os.path.join(basedir, 'images'), '--ImageReader.single_camera', '1'
        ]
        feat_output = (
            subprocess.check_output(
                feature_extractor_args, universal_newlines=True))
        logfile.write(feat_output)
        logger.info('Features extracted done')

        exhaustive_matcher_args = [
            'colmap',
            match_type,
            '--database_path',
            os.path.join(basedir, 'database.db'),
        ]

        match_output = (
            subprocess.check_output(
                exhaustive_matcher_args, universal_newlines=True))
        logfile.write(match_output)
        logger.info('Features matched done')

        p = os.path.join(basedir, 'sparse')
        if not os.path.exists(p):
            os.makedirs(p)

        mapper_args = [
            'colmap',
            'mapper',
            '--database_path',
            os.path.join(basedir, 'database.db'),
            '--image_path',
            os.path.join(basedir, 'images'),
            '--output_path',
            os.path.join(
                basedir, 'sparse'
            ),  # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads',
            '16',
            '--Mapper.init_min_tri_angle',
            '4',
            '--Mapper.multiple_models',
            '0',
            '--Mapper.extract_colors',
            '0',
        ]

        map_output = (
            subprocess.check_output(mapper_args, universal_newlines=True))
        logfile.write(map_output)
        logger.info('Sparse map created done.')

        bundle_adjuster_cmd = [
            'colmap',
            'bundle_adjuster',
            '--input_path',
            os.path.join(basedir, 'sparse/0'),
            '--output_path',
            os.path.join(basedir, 'sparse/0'),
            '--BundleAdjustment.refine_principal_point',
            '1',
        ]
        map_output = (
            subprocess.check_output(
                bundle_adjuster_cmd, universal_newlines=True))
        logfile.write(map_output)
        logger.info('Refining intrinsics done.')

        if use_distortion:
            os.makedirs(os.path.join(basedir, 'preprocess'), exist_ok=True)
            distort_cmd = [
                'colmap', 'image_undistorter', '--image_path',
                os.path.join(basedir, 'images'), '--input_path',
                os.path.join(basedir, 'sparse/0'), '--output_path',
                os.path.join(basedir, 'preprocess'), '--output_type', 'COLMAP'
            ]
            map_output = (
                subprocess.check_output(distort_cmd, universal_newlines=True))
            logfile.write(map_output)
            logger.info('Image distortion done.')

        logfile.close()
        logger.info(
            'Finished running COLMAP, see {} for logs'.format(logfile_name))

    def gen_poses(self, basedir, match_type, use_distortion):
        files_needed = [
            '{}.bin'.format(f) for f in ['cameras', 'images', 'points3D']
        ]
        if os.path.exists(os.path.join(basedir, 'sparse/0')):
            files_had = os.listdir(os.path.join(basedir, 'sparse/0'))
        else:
            files_had = []
        if not all([f in files_had for f in files_needed]):
            logger.info('Need to run COLMAP')
            self.run_colmap(basedir, match_type, use_distortion)
        else:
            logger.info('Don\'t need to run COLMAP')
