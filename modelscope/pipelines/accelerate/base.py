import os.path
from abc import abstractmethod
from typing import List, Union

import torch.cuda

from modelscope import read_config, snapshot_download
from modelscope.utils.config import Config


class InferFramework:

    def __init__(self, model_id_or_dir: str, **kwargs):
        """
        Args:
            model_id_or_dir(`str`): The model id of the modelhub or a local dir containing model files.
        """
        if os.path.exists(model_id_or_dir):
            self.model_dir = model_id_or_dir
        else:
            self.model_dir = snapshot_download(model_id_or_dir)

        model_supported = self.model_type_supported(model_id_or_dir)
        config: Config = read_config(self.model_dir)
        model_type = config.safe_get('model.type')
        if model_type is not None:
            model_supported = model_supported or self.model_type_supported(
                model_type)
        config_file = os.path.join(self.model_dir, 'config.json')
        if os.path.isfile(config_file):
            config = Config.from_file(config_file)
            model_type = config.safe_get('model_type')
            if model_type is not None:
                model_supported = model_supported or self.model_type_supported(
                    model_type)

        if not model_supported:
            raise ValueError(
                f'Model accelerating not supported: {model_id_or_dir}')

    @abstractmethod
    def __call__(self, prompts: Union[List[str], List[List[int]]],
                 **kwargs) -> List[str]:
        """
        Args:
            prompts(`Union[List[str], List[List[int]]]`):
                The string batch or the token list batch to input to the model.
        Returns:
            The answers in list according to the input prompt batch.
        """
        pass

    def model_type_supported(self, model_type: str):
        return False

    @staticmethod
    def check_gpu_compatibility(major_version: int):
        """Check the GPU compatibility.
        """
        major, _ = torch.cuda.get_device_capability()
        return major >= major_version

    @classmethod
    def from_pretrained(cls, model_id_or_dir, framework='vllm', **kwargs):
        """Instantiate the model wrapped by an accelerate framework.
        Args:
            model_id_or_dir(`str`): The model id of the modelhub or a local dir containing model files.
            framework(`str`): The framework to use.
        Returns:
            The wrapped model.
        """
        if framework == 'vllm':
            from .vllm import Vllm
            vllm = Vllm(model_id_or_dir, **kwargs)
            vllm.llm_framework = framework
            return vllm
        else:
            raise ValueError(f'Framework not supported: {framework}')
