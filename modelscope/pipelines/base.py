# Copyright (c) Alibaba, Inc. and its affiliates.

import os.path as osp
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Union

from maas_hub.snapshot_download import snapshot_download

from modelscope.models.base import Model
from modelscope.preprocessors import Preprocessor
from modelscope.pydatasets import PyDataset
from modelscope.utils.config import Config
from modelscope.utils.hub import get_model_cache_dir
from .util import is_model_name

Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[str, PyDataset, 'PIL.Image.Image', 'numpy.ndarray']
InputModel = Union[str, Model]

output_keys = [
]  # 对于不同task的pipeline，规定标准化的输出key，用以对接postprocess,同时也用来标准化postprocess后输出的key


class Pipeline(ABC):

    def initiate_single_model(self, model):
        if isinstance(model, str):
            if not osp.exists(model):
                cache_path = get_model_cache_dir(model)
                model = cache_path if osp.exists(
                    cache_path) else snapshot_download(model)
            return Model.from_pretrained(model) if is_model_name(
                model) else model
        elif isinstance(model, Model):
            return model
        else:
            if model:
                raise ValueError(
                    f'model type for single model is either str or Model, but got type {type(model)}'
                )

    def initiate_multiple_models(self, input_models: List[InputModel]):
        models = []
        for model in input_models:
            models.append(self.initiate_single_model(model))
        return models

    def __init__(self,
                 config_file: str = None,
                 model: Union[InputModel, List[InputModel]] = None,
                 preprocessor: Union[Preprocessor, List[Preprocessor]] = None,
                 **kwargs):
        """ Base class for pipeline.

        If config_file is provided, model and preprocessor will be
        instantiated from corresponding config. Otherwise, model
        and preprocessor will be constructed separately.

        Args:
            config_file(str, optional): Filepath to configuration file.
            model: (list of) Model name or model object
            preprocessor: (list of) Preprocessor object
        """
        if config_file is not None:
            self.cfg = Config.from_file(config_file)
        if not isinstance(model, List):
            self.model = self.initiate_single_model(model)
            self.models = [self.model]
        else:
            self.models = self.initiate_multiple_models(model)

        self.has_multiple_models = len(self.models) > 1
        self.preprocessor = preprocessor

    def __call__(self, input: Union[Input, List[Input]], *args,
                 **post_kwargs) -> Union[Dict[str, Any], Generator]:
        # model provider should leave it as it is
        # modelscope library developer will handle this function

        # simple showcase, need to support iterator type for both tensorflow and pytorch
        # input_dict = self._handle_input(input)
        if isinstance(input, list):
            output = []
            for ele in input:
                output.append(self._process_single(ele, *args, **post_kwargs))

        elif isinstance(input, PyDataset):
            return self._process_iterator(input, *args, **post_kwargs)

        else:
            output = self._process_single(input, *args, **post_kwargs)
        return output

    def _process_iterator(self, input: Input, *args, **post_kwargs):
        for ele in input:
            yield self._process_single(ele, *args, **post_kwargs)

    def _process_single(self, input: Input, *args,
                        **post_kwargs) -> Dict[str, Any]:
        out = self.preprocess(input)
        out = self.forward(out)
        out = self.postprocess(out, **post_kwargs)
        return out

    def preprocess(self, inputs: Input) -> Dict[str, Any]:
        """ Provide default implementation based on preprocess_cfg and user can reimplement it
        """
        assert self.preprocessor is not None, 'preprocess method should be implemented'
        assert not isinstance(self.preprocessor, List),\
            'default implementation does not support using multiple preprocessors.'
        return self.preprocessor(inputs)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Provide default implementation using self.model and user can reimplement it
        """
        assert self.model is not None, 'forward method should be implemented'
        assert not self.has_multiple_models, 'default implementation does not support multiple models in a pipeline.'
        return self.model(inputs)

    @abstractmethod
    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError('postprocess')
