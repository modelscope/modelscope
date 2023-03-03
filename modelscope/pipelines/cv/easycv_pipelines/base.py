# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import os.path as osp
from typing import Any

import numpy as np
from easycv.utils.ms_utils import EasyCVMeta
from PIL import ImageFile

from modelscope.hub.snapshot_download import snapshot_download
from modelscope.pipelines.util import is_official_hub_path
from modelscope.utils.config import Config
from modelscope.utils.constant import (DEFAULT_MODEL_REVISION, Invoke,
                                       ModelFile, ThirdParty)
from modelscope.utils.device import create_device


class EasyCVPipeline(object):
    """Base pipeline for EasyCV.
    Loading configuration file of modelscope style by default,
    but it is actually use the predictor api of easycv to predict.
    So here we do some adaptation work for configuration and predict api.
    """

    def __init__(self, model: str, model_file_pattern='*.pt', *args, **kwargs):
        """
            model (str): model id on modelscope hub or local model path.
            model_file_pattern (str): model file pattern.

        """
        self.model_file_pattern = model_file_pattern

        assert isinstance(model, str)
        if osp.exists(model):
            model_dir = model
        else:
            assert is_official_hub_path(
                model), 'Only support local model path and official hub path!'
            model_dir = snapshot_download(
                model_id=model,
                revision=DEFAULT_MODEL_REVISION,
                user_agent={
                    Invoke.KEY: Invoke.PIPELINE,
                    ThirdParty.KEY: ThirdParty.EASYCV
                })

        assert osp.isdir(model_dir)
        model_files = glob.glob(
            os.path.join(model_dir, self.model_file_pattern))
        assert len(
            model_files
        ) == 1, f'Need one model file, but find {len(model_files)}: {model_files}'

        model_path = model_files[0]
        self.model_path = model_path
        self.model_dir = model_dir

        # get configuration file from source model dir
        self.config_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
        assert os.path.exists(
            self.config_file
        ), f'Not find "{ModelFile.CONFIGURATION}" in model directory!'

        self.cfg = Config.from_file(self.config_file)
        if 'device' in kwargs:
            kwargs['device'] = create_device(kwargs['device'])
        if 'predictor_config' in kwargs:
            kwargs.pop('predictor_config')
        self.predict_op = self._build_predict_op(**kwargs)

    def _build_predict_op(self, **kwargs):
        """Build EasyCV predictor."""
        from easycv.predictors.builder import build_predictor

        easycv_config = self._to_easycv_config()
        pipeline_op = build_predictor(self.cfg.pipeline.predictor_config, {
            'model_path': self.model_path,
            'config_file': easycv_config,
            **kwargs
        })
        return pipeline_op

    def _to_easycv_config(self):
        """Adapt to EasyCV predictor."""
        # TODO: refine config compatibility problems

        easycv_arch = self.cfg.model.pop(EasyCVMeta.ARCH, None)
        model_cfg = self.cfg.model
        # Revert to the configuration of easycv
        if easycv_arch is not None:
            model_cfg.update(easycv_arch)

        easycv_config = Config(dict(model=model_cfg))

        reserved_keys = []
        if hasattr(self.cfg, EasyCVMeta.META):
            easycv_meta_cfg = getattr(self.cfg, EasyCVMeta.META)
            reserved_keys = easycv_meta_cfg.get(EasyCVMeta.RESERVED_KEYS, [])
            for key in reserved_keys:
                easycv_config.merge_from_dict({key: getattr(self.cfg, key)})
        if 'test_pipeline' not in reserved_keys:
            easycv_config.merge_from_dict(
                {'test_pipeline': self.cfg.dataset.val.get('pipeline', [])})

        return easycv_config

    def _is_single_inputs(self, inputs):
        if isinstance(inputs, str) or (isinstance(inputs, list)
                                       and len(inputs) == 1) or isinstance(
                                           inputs, np.ndarray) or isinstance(
                                               inputs, ImageFile.ImageFile):
            return True

        return False

    def __call__(self, inputs) -> Any:
        outputs = self.predict_op(inputs)

        if self._is_single_inputs(inputs):
            outputs = outputs[0]

        return outputs
