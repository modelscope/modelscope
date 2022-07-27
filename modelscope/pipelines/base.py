# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from abc import ABC, abstractmethod
from contextlib import contextmanager
from threading import Lock
from typing import Any, Dict, Generator, List, Mapping, Union

import numpy as np

from modelscope.models.base import Model
from modelscope.msdatasets import MsDataset
from modelscope.outputs import TASK_OUTPUTS
from modelscope.preprocessors import Preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import Frameworks, ModelFile
from modelscope.utils.import_utils import is_tf_available, is_torch_available
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import create_device
from .util import is_model, is_official_hub_path

if is_torch_available():
    import torch

if is_tf_available():
    import tensorflow as tf

Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[str, tuple, MsDataset, 'Image.Image', 'numpy.ndarray']
InputModel = Union[str, Model]

logger = get_logger()


class Pipeline(ABC):

    def initiate_single_model(self, model):
        if isinstance(model, str):
            logger.info(f'initiate model from {model}')
        if isinstance(model, str) and is_official_hub_path(model):
            logger.info(f'initiate model from location {model}.')
            # expecting model has been prefetched to local cache beforehand
            return Model.from_pretrained(
                model, model_prefetched=True) if is_model(model) else model
        elif isinstance(model, Model):
            return model
        else:
            if model and not isinstance(model, str):
                raise ValueError(
                    f'model type for single model is either str or Model, but got type {type(model)}'
                )
            return model

    def initiate_multiple_models(self, input_models: List[InputModel]):
        models = []
        for model in input_models:
            models.append(self.initiate_single_model(model))
        return models

    def __init__(self,
                 config_file: str = None,
                 model: Union[InputModel, List[InputModel]] = None,
                 preprocessor: Union[Preprocessor, List[Preprocessor]] = None,
                 device: str = 'gpu',
                 **kwargs):
        """ Base class for pipeline.

        If config_file is provided, model and preprocessor will be
        instantiated from corresponding config. Otherwise, model
        and preprocessor will be constructed separately.

        Args:
            config_file(str, optional): Filepath to configuration file.
            model: (list of) Model name or model object
            preprocessor: (list of) Preprocessor object
            device (str): gpu device or cpu device to use
        """
        if config_file is not None:
            self.cfg = Config.from_file(config_file)
        if not isinstance(model, List):
            self.model = self.initiate_single_model(model)
            self.models = [self.model]
        else:
            self.model = None
            self.models = self.initiate_multiple_models(model)

        self.has_multiple_models = len(self.models) > 1
        self.preprocessor = preprocessor

        if self.model or (self.has_multiple_models and self.models[0]):
            self.framework = self._get_framework()
        else:
            self.framework = None

        assert device in ['gpu', 'cpu'], 'device should be either cpu or gpu.'
        self.device_name = device
        if self.framework == Frameworks.torch:
            self.device = create_device(self.device_name == 'cpu')
        self._model_prepare = False
        self._model_prepare_lock = Lock()

    def prepare_model(self):
        self._model_prepare_lock.acquire(timeout=600)

        def _prepare_single(model):
            if isinstance(model, torch.nn.Module):
                model.to(self.device)
            elif hasattr(model, 'model') and isinstance(
                    model.model, torch.nn.Module):
                model.model.to(self.device)

        if not self._model_prepare:
            # prepare model for pytorch
            if self.framework == Frameworks.torch:
                if self.has_multiple_models:
                    for m in self.models:
                        _prepare_single(m)
                else:
                    _prepare_single(self.model)
            self._model_prepare = True
        self._model_prepare_lock.release()

    @contextmanager
    def place_device(self):
        """ device placement function, allow user to specify which device to place pipeline

        Returns:
            Context manager

        Examples:

        ```python
        # Requests for using pipeline on cuda:0 for gpu
        pipeline = pipeline(..., device='gpu')
        with pipeline.device():
            output = pipe(...)
        ```
        """
        if self.framework == Frameworks.tf:
            if self.device_name == 'cpu':
                with tf.device('/CPU:0'):
                    yield
            else:
                with tf.device('/device:GPU:0'):
                    yield

        elif self.framework == Frameworks.torch:
            if self.device_name == 'gpu':
                device = create_device()
                if device.type == 'gpu':
                    torch.cuda.set_device(device)
            yield
        else:
            yield

    def _get_framework(self) -> str:
        frameworks = []
        for m in self.models:
            if isinstance(m, Model):
                model_dir = m.model_dir
            else:
                assert isinstance(m,
                                  str), 'model should be either str or Model.'
                model_dir = m
            cfg_file = osp.join(model_dir, ModelFile.CONFIGURATION)
            cfg = Config.from_file(cfg_file)
            frameworks.append(cfg.framework)
        if not all(x == frameworks[0] for x in frameworks):
            raise ValueError(
                f'got multiple models, but they are in different frameworks {frameworks}'
            )

        return frameworks[0]

    def __call__(self, input: Union[Input, List[Input]], *args,
                 **kwargs) -> Union[Dict[str, Any], Generator]:
        # model provider should leave it as it is
        # modelscope library developer will handle this function

        # place model to cpu or gpu
        if (self.model or (self.has_multiple_models and self.models[0])):
            if not self._model_prepare:
                self.prepare_model()

        # simple showcase, need to support iterator type for both tensorflow and pytorch
        # input_dict = self._handle_input(input)

        # sanitize the parameters
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs)
        kwargs['preprocess_params'] = preprocess_params
        kwargs['forward_params'] = forward_params
        kwargs['postprocess_params'] = postprocess_params

        if isinstance(input, list):
            output = []
            for ele in input:
                output.append(self._process_single(ele, *args, **kwargs))

        elif isinstance(input, MsDataset):
            return self._process_iterator(input, *args, **kwargs)

        else:
            output = self._process_single(input, *args, **kwargs)
        return output

    def _sanitize_parameters(self, **pipeline_parameters):
        """
        this method should sanitize the keyword args to preprocessor params,
        forward params and postprocess params on '__call__' or '_process_single' method
        considered to be a normal classmethod with default implementation / output

        Default Returns:
            Dict[str, str]:  preprocess_params = {}
            Dict[str, str]:  forward_params = {}
            Dict[str, str]:  postprocess_params = pipeline_parameters
        """
        return {}, {}, pipeline_parameters

    def _process_iterator(self, input: Input, *args, **kwargs):
        for ele in input:
            yield self._process_single(ele, *args, **kwargs)

    def _collate_fn(self, data):
        """Prepare the input just before the forward function.
        This method will move the tensors to the right device.
        Usually this method does not need to be overridden.

        Args:
            data: The data out of the dataloader.

        Returns: The processed data.

        """
        from torch.utils.data.dataloader import default_collate
        from modelscope.preprocessors import InputFeatures
        if isinstance(data, dict) or isinstance(data, Mapping):
            return type(data)(
                {k: self._collate_fn(v)
                 for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            if isinstance(data[0], (int, float)):
                return default_collate(data).to(self.device)
            else:
                return type(data)(self._collate_fn(v) for v in data)
        elif isinstance(data, np.ndarray):
            if data.dtype.type is np.str_:
                return data
            else:
                return self._collate_fn(torch.from_numpy(data))
        elif isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif isinstance(data, InputFeatures):
            return data
        else:
            raise ValueError(f'Unsupported data type {type(data)}')

    def _process_single(self, input: Input, *args, **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params')
        forward_params = kwargs.get('forward_params')
        postprocess_params = kwargs.get('postprocess_params')

        out = self.preprocess(input, **preprocess_params)
        with self.place_device():
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    out = self._collate_fn(out)
                    out = self.forward(out, **forward_params)
            else:
                out = self.forward(out, **forward_params)

        out = self.postprocess(out, **postprocess_params)
        self._check_output(out)
        return out

    def _check_output(self, input):
        # this attribute is dynamically attached by registry
        # when cls is registered in registry using task name
        task_name = self.group_key
        if task_name not in TASK_OUTPUTS:
            logger.warning(f'task {task_name} output keys are missing')
            return
        output_keys = TASK_OUTPUTS[task_name]
        missing_keys = []
        for k in output_keys:
            if k not in input:
                missing_keys.append(k)
        if len(missing_keys) > 0:
            raise ValueError(f'expected output keys are {output_keys}, '
                             f'those {missing_keys} are missing')

    def preprocess(self, inputs: Input, **preprocess_params) -> Dict[str, Any]:
        """ Provide default implementation based on preprocess_cfg and user can reimplement it
        """
        assert self.preprocessor is not None, 'preprocess method should be implemented'
        assert not isinstance(self.preprocessor, List),\
            'default implementation does not support using multiple preprocessors.'
        return self.preprocessor(inputs, **preprocess_params)

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        """ Provide default implementation using self.model and user can reimplement it
        """
        assert self.model is not None, 'forward method should be implemented'
        assert not self.has_multiple_models, 'default implementation does not support multiple models in a pipeline.'
        return self.model(inputs, **forward_params)

    @abstractmethod
    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        raise NotImplementedError('postprocess')
