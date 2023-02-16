# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from typing import Any, Dict

import torch

from modelscope.metainfo import Models
from modelscope.models.base.base_torch_model import TorchModel
from modelscope.models.builder import MODELS
from modelscope.msdatasets import MsDataset
from modelscope.utils.config import Config
from modelscope.utils.constant import DownloadMode, ModelFile, Tasks
from modelscope.utils.logger import get_logger
from .models.calibration_layer import PrototypicalCalibrationBlock
from .models.defrcn import DeFRCN
from .utils.configuration_mapper import CfgMapper
from .utils.register_data import register_data
from .utils.requirements_check import requires_version

logger = get_logger()
__all__ = ['DeFRCNForFewShot']


@MODELS.register_module(
    Tasks.image_fewshot_detection, module_name=Models.defrcn)
class DeFRCNForFewShot(TorchModel):
    """ Few-shot object detection model DeFRCN. The model requires detectron2-0.3 and pytorch-1.11.
        Model config params mainly from detectron2, you can use detectron2 config file to initialize model.
        Detail configs can be visited on detectron2.config.defaults and .models.defaults_config.
    """

    def __init__(self,
                 model_dir: str,
                 _cfg_dict: Config = None,
                 *args,
                 **kwargs):
        """initialize the few-shot defrcn model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            _cfg_dict (Config): An optional model config. If provided, it will replace
                the config read out of the `model_name_or_path`
        """
        requires_version()

        super().__init__(model_dir, *args, **kwargs)

        if _cfg_dict is None:
            self.config = Config.from_file(
                os.path.join(model_dir, ModelFile.CONFIGURATION))
        else:
            self.config = _cfg_dict

        self.model_cfg = CfgMapper(self.config).__call__()

        data_dir = self.config.safe_get('datasets.root', None)
        data_type = self.config.safe_get('datasets.type', 'pascal_voc')

        if self.training or self.model_cfg.TEST.PCB_ENABLE:
            if data_dir is None:  # use default datasets
                dataset_name = 'VOC_fewshot' if data_type == 'pascal_voc' else 'coco2014_fewshot'
                logger.warning('data_dir is none, use default {} data.'.format(
                    dataset_name))
                data_voc = MsDataset.load(
                    dataset_name=dataset_name,
                    namespace='shimin2023',
                    split='train',
                    download_mode=DownloadMode.REUSE_DATASET_IF_EXISTS)
                data_dir = os.path.join(
                    data_voc.config_kwargs['split_config']['train'], 'data')
                logger.info('{} datasets download dir is {}'.format(
                    dataset_name, data_dir))
            register_data(data_type, data_dir)

        self.model = DeFRCN(self.model_cfg)

        if self.model_cfg.TEST.PCB_ENABLE:
            if not os.path.exists(self.model_cfg.TEST.PCB_MODELPATH):
                logger.warning('{} no model.'.format(
                    self.model_cfg.TEST.PCB_MODELPATH))
                self.model_cfg.TEST.PCB_MODELPATH = os.path.join(
                    model_dir,
                    'ImageNetPretrained/torchvision/resnet101-5d3b4d8f.pth')
                logger.info('PCB use default model {}'.format(
                    self.model_cfg.TEST.PCB_MODELPATH))
            self.pcb = PrototypicalCalibrationBlock(self.model_cfg)

        self.model_cfg.freeze()

    def forward(self, inputs) -> Any:
        """return the result by the model

        Args:
            inputs (list): the preprocessed data

        Returns:
            Any: results
        """
        if self.training:
            return self.model.forward(inputs)
        else:
            return self.model.inference(inputs)

    def inference(self, input: Dict[str, Any]) -> Any:
        with torch.no_grad():
            results = self.model([input])
            if self.model_cfg.TEST.PCB_ENABLE:
                results = self.pcb.execute_calibration([input], results)
        return results[0] if len(results) > 0 else None

    def get_model_cfg(self):
        return self.model_cfg
