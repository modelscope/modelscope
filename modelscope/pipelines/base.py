# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import os.path as osp
import random
from abc import ABC, abstractmethod
from functools import partial
from multiprocessing import Pool
from threading import Lock
from typing import Any, Dict, Generator, List, Mapping, Union

import numpy as np
from packaging import version

from modelscope.models.base import Model
from modelscope.msdatasets import MsDataset
from modelscope.outputs import TASK_OUTPUTS, ModelOutputBase
from modelscope.pipeline_inputs import TASK_INPUTS, check_input_type
from modelscope.preprocessors import Preprocessor
from modelscope.utils.config import Config
from modelscope.utils.constant import Frameworks, Invoke, ModelFile
from modelscope.utils.device import (create_device, device_placement,
                                     verify_device)
from modelscope.utils.hub import read_config, snapshot_download
from modelscope.utils.import_utils import is_tf_available, is_torch_available
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import compile_model
from .util import is_model, is_official_hub_path

if is_torch_available():
    import torch

if is_tf_available():
    pass

Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[str, tuple, MsDataset, 'Image.Image', 'numpy.ndarray']
InputModel = Union[str, Model, 'torch.nn.Module']

logger = get_logger()


class Pipeline(ABC):
    """Pipeline base.
    """

    def initiate_single_model(self, model, **kwargs):
        if isinstance(model, str):
            logger.info(f'initiate model from {model}')
        if isinstance(model, str) and is_official_hub_path(model):
            logger.info(f'initiate model from location {model}.')
            # expecting model has been prefetched to local cache beforehand
            return Model.from_pretrained(
                model,
                device=self.device_name,
                model_prefetched=True,
                invoked_by=Invoke.PIPELINE,
                device_map=self.device_map,
                **kwargs) if is_model(model) else model
        else:
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
                 auto_collate=True,
                 device_map=None,
                 **kwargs):
        """ Base class for pipeline.

        If config_file is provided, model and preprocessor will be
        instantiated from corresponding config. Otherwise, model
        and preprocessor will be constructed separately.

        Args:
            config_file(str, optional): Filepath to configuration file.
            model: (list of) Model name or model object
            preprocessor: (list of) Preprocessor object
            device (str): device str, should be either cpu, cuda, gpu, gpu:X or cuda:X
            auto_collate (bool): automatically to convert data to tensor or not.
            compile (bool, optional): Compile the model with torch 2.0, default False
            compile_options (dict, optional): The compile options if compile=True,
                default None to use the default params of 'TorchModel.compile'.
        """
        if device_map is not None:
            assert device == 'gpu', '`device` and `device_map` cannot be input at the same time!'
        self.device_map = device_map
        verify_device(device)
        self.device_name = device

        if not isinstance(model, List):
            self.model = self.initiate_single_model(model, **kwargs)
            self.models = [self.model]
        else:
            self.model = None
            self.models = self.initiate_multiple_models(model)

        self.has_multiple_models = len(self.models) > 1

        if config_file is not None:
            self.cfg = Config.from_file(config_file)
            model_dir = os.path.dirname(config_file)
        elif not self.has_multiple_models:
            if isinstance(self.model, str):
                model_dir = self.model
            else:
                model_dir = self.model.model_dir
            self.cfg = read_config(model_dir)

        if preprocessor is None and not self.has_multiple_models:
            self.preprocessor = Preprocessor.from_pretrained(model_dir)
        else:
            self.preprocessor = preprocessor

        if self.model or (self.has_multiple_models and self.models[0]):
            self.framework = self._get_framework()
        else:
            self.framework = None

        if self.framework == Frameworks.torch:
            self.device = create_device(self.device_name)
        self._model_prepare = False
        self._model_prepare_lock = Lock()
        self._auto_collate = auto_collate
        self._compile = kwargs.get('compile', False)
        self._compile_options = kwargs.get('compile_options', {})

    def prepare_model(self):
        """ Place model on certain device for pytorch models before first inference
        """
        self._model_prepare_lock.acquire(timeout=600)

        def _prepare_single(model):
            if not isinstance(model, torch.nn.Module) and hasattr(
                    model, 'model'):
                model = model.model
            if not isinstance(model, torch.nn.Module):
                return
            model.eval()
            from modelscope.utils.torch_utils import is_on_same_device
            if is_on_same_device(model):
                model.to(self.device)

        if not self._model_prepare:
            # prepare model for pytorch
            if self.framework == Frameworks.torch:
                if self.has_multiple_models:
                    for m in self.models:
                        _prepare_single(m)
                    if self._compile:
                        self.models = [
                            compile_model(m, **self._compile_options)
                            for m in self.models
                        ]
                else:
                    _prepare_single(self.model)
                    if self._compile:
                        self.model = compile_model(self.model,
                                                   **self._compile_options)
            self._model_prepare = True
        self._model_prepare_lock.release()

    def _get_framework(self) -> str:
        frameworks = []
        for m in self.models:
            if isinstance(m, str):
                model_dir = m
            else:
                model_dir = m.model_dir
            cfg_file = osp.join(model_dir, ModelFile.CONFIGURATION)
            cfg = Config.from_file(cfg_file)
            frameworks.append(cfg.framework)
        if not all(x == frameworks[0] for x in frameworks):
            logger.warning(
                f'got multiple models, but they are in different frameworks {frameworks}'
            )
            return None

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
        batch_size = kwargs.pop('batch_size', None)
        preprocess_params, forward_params, postprocess_params = self._sanitize_parameters(
            **kwargs)
        kwargs['preprocess_params'] = preprocess_params
        kwargs['forward_params'] = forward_params
        kwargs['postprocess_params'] = postprocess_params
        if isinstance(input, list):
            if batch_size is None:
                output = []
                for ele in input:
                    output.append(self._process_single(ele, *args, **kwargs))
            else:
                output = self._process_batch(input, batch_size, **kwargs)

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
        return collate_fn(data, self.device)

    def _process_single(self, input: Input, *args, **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params', {})
        forward_params = kwargs.get('forward_params', {})
        postprocess_params = kwargs.get('postprocess_params', {})
        self._check_input(input)
        out = self.preprocess(input, **preprocess_params)

        with device_placement(self.framework, self.device_name):
            if self.framework == Frameworks.torch:
                with torch.no_grad():
                    if self._auto_collate:
                        out = self._collate_fn(out)
                    out = self.forward(out, **forward_params)
            else:
                out = self.forward(out, **forward_params)

        out = self.postprocess(out, **postprocess_params)
        self._check_output(out)
        return out

    def _batch(self, data_list):
        batch_data = {}
        for sample_preprocessed in data_list:
            for k, v in sample_preprocessed.items():
                value_list = batch_data.get(k, [])
                value_list.append(v)
                batch_data[k] = value_list
        for k in batch_data.keys():
            if isinstance(batch_data[k][0], torch.Tensor):
                batch_data[k] = torch.cat(batch_data[k])
        return batch_data

    def _process_batch(self, input: List[Input], batch_size,
                       **kwargs) -> Dict[str, Any]:
        preprocess_params = kwargs.get('preprocess_params')
        forward_params = kwargs.get('forward_params')
        postprocess_params = kwargs.get('postprocess_params')

        # batch data
        output_list = []
        for i in range(0, len(input), batch_size):
            end = min(i + batch_size, len(input))
            real_batch_size = end - i
            preprocessed_list = [
                self.preprocess(i, **preprocess_params) for i in input[i:end]
            ]

            with device_placement(self.framework, self.device_name):
                if self.framework == Frameworks.torch:
                    with torch.no_grad():
                        batched_out = self._batch(preprocessed_list)
                        if self._auto_collate:
                            batched_out = self._collate_fn(batched_out)
                        batched_out = self.forward(batched_out,
                                                   **forward_params)
                else:
                    batched_out = self._batch(preprocessed_list)
                    batched_out = self.forward(batched_out, **forward_params)

            for batch_idx in range(real_batch_size):
                out = {}
                for k, element in batched_out.items():
                    if element is not None:
                        if isinstance(element, (tuple, list)):
                            if isinstance(element[0], torch.Tensor):
                                out[k] = type(element)(
                                    e[batch_idx:batch_idx + 1]
                                    for e in element)
                            else:
                                # Compatible with traditional pipelines
                                out[k] = element[batch_idx]
                        else:
                            out[k] = element[batch_idx:batch_idx + 1]
                out = self.postprocess(out, **postprocess_params)
                self._check_output(out)
                output_list.append(out)

        return output_list

    def _check_input(self, input):
        task_name = self.group_key
        if task_name in TASK_INPUTS:
            input_type = TASK_INPUTS[task_name]

            # if multiple input formats are defined, we first
            # found the one that match input data and check
            if isinstance(input_type, list):
                matched_type = None
                for t in input_type:
                    if isinstance(input, (dict, tuple)):
                        if type(t) == type(input):
                            matched_type = t
                            break
                    elif isinstance(t, str):
                        matched_type = t
                        break
                if matched_type is None:
                    err_msg = 'input data format for current pipeline should be one of following: \n'
                    for t in input_type:
                        err_msg += f'{t}\n'
                    raise ValueError(err_msg)
                else:
                    input_type = matched_type

            if isinstance(input_type, str):
                check_input_type(input_type, input)
            elif isinstance(input_type, tuple):
                assert isinstance(input, tuple), 'input should be a tuple'
                for t, input_ele in zip(input_type, input):
                    check_input_type(t, input_ele)
            elif isinstance(input_type, dict):
                for k in input_type.keys():
                    # allow single input for multi-modal models
                    if isinstance(input, dict) and k in input:
                        check_input_type(input_type[k], input[k])
            else:
                raise ValueError(f'invalid input_type definition {input_type}')
        elif not getattr(self, '_input_has_warned', False):
            logger.warning(f'task {task_name} input definition is missing')
            self._input_has_warned = True

    def _check_output(self, input):
        # this attribute is dynamically attached by registry
        # when cls is registered in registry using task name
        task_name = self.group_key
        if task_name not in TASK_OUTPUTS:
            if not getattr(self, '_output_has_warned', False):
                logger.warning(f'task {task_name} output keys are missing')
                self._output_has_warned = True
            return
        output_keys = TASK_OUTPUTS[task_name]
        missing_keys = []
        input = input.keys() if isinstance(input,
                                           (dict, ModelOutputBase)) else input
        for k in output_keys:
            if isinstance(k, (dict, ModelOutputBase)) and k not in input:
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

    def postprocess(self, inputs: Dict[str, Any],
                    **post_params) -> Dict[str, Any]:
        """ If current pipeline support model reuse, common postprocess
            code should be write here.

        Args:
            inputs:  input data
            post_params:   post process parameters

        Return:
            dict of results:  a dict containing outputs of model, each
                output should have the standard output name.
        """
        raise NotImplementedError('postprocess')


class DistributedPipeline(Pipeline):
    """This pipeline is used to load multi gpu models.

    What will this class do:
    1. Read the global config from the configuration.json
    2. Set the multiprocessing method to spawn
    3. Open a multiprocessing pool of the world_size to instantiate model pieces.
    4. Set the master port and ip
    5. Call _instantiate_one to instantiate one model piece,
    This method should be implemented by the derived class.
    6. After the forward method is called, do preprocess in main process and
    call _forward_one to collect results, and do post process in main process.

    NOTE: _instantiate_one and _forward_one are class methods, any derived class should implement them and
    store the model handler in the class field.
    """

    def __init__(self,
                 model: str = None,
                 preprocessor: Union[Preprocessor, List[Preprocessor]] = None,
                 auto_collate=True,
                 **kwargs):
        # DistributedPipeline uses classmethod to initialize model
        # without calling super().__init__ method
        self.preprocessor = preprocessor
        self._model_prepare = False
        self._model_prepare_lock = Lock()
        self._auto_collate = auto_collate

        if os.path.exists(model):
            self.model_dir = model
        else:
            self.model_dir = snapshot_download(model)
        self.cfg = read_config(self.model_dir)
        self.world_size = self._get_world_size(self.cfg)
        self.model_pool = None
        self.device_name = 'cpu'
        self.device = create_device(self.device_name)
        self.has_multiple_models = False
        self.framework = self.cfg.framework
        torch.multiprocessing.set_start_method('spawn', force=True)

        ranks = list(range(self.world_size))
        self.model_pool = Pool(self.world_size)

        if 'master_ip' not in kwargs:
            kwargs['master_ip'] = '127.0.0.1'
        master_port = int(kwargs['master_port']
                          ) if 'master_port' in kwargs else random.randint(
                              29500, 39500)
        from modelscope.utils.torch_utils import _find_free_port, _is_free_port
        if not _is_free_port(master_port):
            master_port = _find_free_port()
        kwargs['master_port'] = str(master_port)
        # TODO: Pass ip and port to megatron_util for initialization
        os.environ['MASTER_ADDR'] = kwargs['master_ip']
        os.environ['MASTER_PORT'] = kwargs['master_port']

        self.model_pool.map(
            partial(
                self.__class__._instantiate_one,
                model_dir=self.model_dir,
                **self.cfg.model,
                **kwargs), ranks)
        self.models = []

    def __del__(self):
        if hasattr(self, 'model_pool') and self.model_pool is not None:
            self.model_pool.terminate()

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['model_pool']
        del self_dict['preprocessor']
        del self_dict['_model_prepare_lock']
        return self_dict

    @classmethod
    def _instantiate_one(cls, rank, model_dir, **kwargs):
        """Instantiate one model piece.

        Args:
            rank: The model rank.
            model_dir: The model_dir in the node.
            kwargs: Any extra args.

        Returns:
            None. The model handler should be kept in the class field.
        """
        pass

    def forward(self, inputs: Dict[str, Any],
                **forward_params) -> Dict[str, Any]:
        inputs = {
            'inputs': inputs,
            'forward_params': forward_params,
        }
        res = self.model_pool.map(self.__class__._forward_one,
                                  [inputs] * self.world_size)
        return res[0]

    @classmethod
    def _forward_one(cls, inputs):
        """Forward the inputs to one model piece.

        Use the model handler kept in the class field to forward.

        Args:
            inputs: The inputs after the preprocessing.

        Returns:
            The forward results.
        """
        pass

    def _get_world_size(self, cfg: Config) -> int:
        m_world_size = cfg.safe_get('megatron.world_size')
        if m_world_size is None:
            return cfg.safe_get('model.world_size')
        return m_world_size


def collate_fn(data, device):
    """Prepare the input just before the forward function.
    This method will move the tensors to the right device.
    Usually this method does not need to be overridden.

    Args:
        data: The data out of the dataloader.
        device: The device to move data to.

    Returns: The processed data.

    """
    from torch.utils.data.dataloader import default_collate

    def get_class_name(obj):
        return obj.__class__.__name__

    if isinstance(data, dict) or isinstance(data, Mapping):
        # add compatibility for img_metas for mmlab models
        return type(data)({
            k: collate_fn(v, device) if k != 'img_metas' else v
            for k, v in data.items()
        })
    elif isinstance(data, (tuple, list)):
        if 0 == len(data):
            return torch.Tensor([])
        if isinstance(data[0], (int, float)):
            return default_collate(data).to(device)
        else:
            return type(data)(collate_fn(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        if data.dtype.type is np.str_:
            return data
        else:
            return collate_fn(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (bytes, str, int, float, bool, type(None))):
        return data
    elif get_class_name(data) == 'InputFeatures':
        # modelscope.preprocessors.nlp.InputFeatures
        return data
    elif get_class_name(data) == 'DataContainer':
        # mmcv.parallel.DataContainer
        return data
    else:
        raise ValueError(f'Unsupported data type {type(data)}')
