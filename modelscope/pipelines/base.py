# Copyright (c) Alibaba, Inc. and its affiliates.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

from maas_lib.models import Model
from maas_lib.preprocessors import Preprocessor

Tensor = Union['torch.Tensor', 'tf.Tensor']
Input = Union[str, 'PIL.Image.Image', 'numpy.ndarray']

output_keys = [
]  # 对于不同task的pipeline，规定标准化的输出key，用以对接postprocess,同时也用来标准化postprocess后输出的key


class Pipeline(ABC):

    def __init__(self,
                 config_file: str = None,
                 model: Model = None,
                 preprocessor: Preprocessor = None,
                 **kwargs):
        self.model = model
        self.preprocessor = preprocessor

    def __call__(self, input: Union[Input, List[Input]], *args,
                 **post_kwargs) -> Dict[str, Any]:
        # model provider should leave it as it is
        # maas library developer will handle this function

        # simple showcase, need to support iterator type for both tensorflow and pytorch
        # input_dict = self._handle_input(input)
        if isinstance(input, list):
            output = []
            for ele in input:
                output.append(self._process_single(ele, *args, **post_kwargs))
        else:
            output = self._process_single(input, *args, **post_kwargs)
        return output

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
        return self.preprocessor(inputs)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """ Provide default implementation using self.model and user can reimplement it
        """
        assert self.model is not None, 'forward method should be implemented'
        return self.model(inputs)

    @abstractmethod
    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError('postprocess')
