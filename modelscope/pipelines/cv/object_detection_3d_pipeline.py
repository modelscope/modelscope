# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
from tempfile import TemporaryDirectory
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from modelscope.metainfo import Pipelines
from modelscope.models.cv.object_detection_3d.depe import DepeDetect
from modelscope.models.cv.object_detection_3d.depe.result_vis import \
    plot_result
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from modelscope.pipelines.base import Input, Pipeline
from modelscope.pipelines.builder import PIPELINES
from modelscope.utils.config import Config
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger

logger = get_logger()


@PIPELINES.register_module(
    Tasks.object_detection_3d, module_name=Pipelines.object_detection_3d_depe)
class ObjectDetection3DPipeline(Pipeline):

    def __init__(self, model: str, **kwargs):
        """
        use `model` to create a 3d object detection pipeline for prediction
        Args:
            model: model id on modelscope hub.

        Example:
            >>> import cv2
            >>> from modelscope.pipelines import pipeline
            >>> from modelscope.msdatasets import MsDataset
            >>> ms_ds_nuscenes = MsDataset.load('nuScenes_mini', namespace='shaoxuan')
            >>> data_path = ms_ds_nuscenes.config_kwargs['split_config']
            >>> val_dir = data_path['validation']
            >>> val_root = val_dir + '/' + os.listdir(val_dir)[0] + '/'
            >>> depe = pipeline('object-detection-3d', model='damo/cv_object-detection-3d_depe')
            >>> input_dict = {'data_root': val_root, 'sample_idx': 0}
            >>> result = depe(input_dict)
            >>> cv2.imwrite('result.jpg', result['output_img'])
        """
        super().__init__(model=model, **kwargs)
        config_path = osp.join(model, 'mmcv_depe.py')
        self.cfg = Config.from_file(config_path)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.detector = DepeDetect(model).to(self.device)
        if os.getenv('MODELSCOPE_ENVIRONMENT') == 'eas':
            self.num_workers = 0
        else:
            self.num_workers = 4

    def __call__(self, input, **kwargs):
        """
        Detect 3D objects in images from multi-cameras that passed as inputs

        Args:
            input (`Dict[str, Any]`):
                A dictionary of input consist 2 keys:
                - `data_root` is the path of input data in nuScenes format,
                you can create your own data according steps from model-card,
                if `data_root` is False, a default input data from
                nuScenes-mini validation set will be used, which includes 81
                samples from 2 scenes.
                - `sample_idx` is the index of sample to be inferenced, the
                value should in range of sample number in input data.

        Return:
            A dictionary of result consist 1 keys:
            - `output_img` plots all detection results in one image.

        """
        return super().__call__(input, **kwargs)

    def get_default_data(self):
        ms_ds_nuscenes = MsDataset.load('nuScenes_mini', namespace='shaoxuan')
        data_path = ms_ds_nuscenes.config_kwargs['split_config']
        val_dir = data_path['validation']
        val_root = val_dir + '/' + os.listdir(val_dir)[0] + '/'
        return val_root

    def preprocess(self, input: Input) -> Dict[str, Any]:
        assert 'sample_idx' in input
        idx = input['sample_idx']
        if isinstance(input['sample_idx'], str):
            input['sample_idx'] = int(input['sample_idx'])
        data_root = input.get('data_root', False)
        if data_root is False:
            data_root = self.get_default_data()
            logger.info(f'Note: forward using default data in: {data_root}')
        try:
            if not os.path.exists('/data/Dataset'):
                os.system('mkdir -p /data/Dataset')
            os.system(f'ln -snf {data_root} /data/Dataset/nuScenes')
        except Exception as e:
            raise RuntimeError(
                f'exception:{e}, please make sure to have permission create and write in: /data/Dataset'
            )
        # build the dataloader
        from mmdet3d.datasets import build_dataloader, build_dataset
        self.cfg.data.test.idx_range = (idx, idx + 1)
        self.cfg.data.test.test_mode = True
        self.dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(
            self.dataset,
            samples_per_gpu=1,
            workers_per_gpu=self.num_workers,
            dist=False,
            shuffle=False)
        result = next(iter(data_loader))
        if 'img_metas' in result:
            from mmcv.parallel import scatter
            if next(self.detector.parameters()).is_cuda:
                # scatter to specified GPU
                result = scatter(
                    result, [next(self.detector.parameters()).device.index])[0]
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        with torch.no_grad():
            result = self.detector(**input)
        return result

    def postprocess(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        save_path = kwargs.get('save_path', None)
        if save_path is None:
            save_path = TemporaryDirectory().name
        else:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
        file_path = osp.join(save_path, 'pts_bbox', 'results_nusc.json')
        kwargs_format = {'jsonfile_prefix': save_path}
        self.dataset.format_results(inputs, **kwargs_format)
        logger.info(f'Done, results saved into: {file_path}')
        result_img = plot_result(file_path, vis_thred=0.3)[0]
        return {OutputKeys.OUTPUT_IMG: result_img.astype(np.uint8)}
