import os.path
from abc import abstractmethod
from typing import List

import torch.cuda

from modelscope import snapshot_download


class InferFramework:

    def __init__(self,
                 model_id_or_dir,
                 **kwargs):
        """
        Args:
            model_id_or_dir(`str`): The model id of the modelhub or a local dir containing model files.
        """
        if os.path.exists(model_id_or_dir):
            self.model_dir = model_id_or_dir
        else:
            self.model_dir = snapshot_download(model_id_or_dir)

    @abstractmethod
    def __call__(self, prompts: List[str], **kwargs) -> List[str]:
        """
        Args:
            prompts (`List[str]`): The prompts in batch to generate answers.
        Returns:
            The answers in list according to the input prompt batch.
        """
        pass

    @staticmethod
    def check_gpu_compatibility(major_version: int):
        major, _ = torch.cuda.get_device_capability()
        return major >= major_version

    @classmethod
    def from_pretrained(cls, model_id_or_dir, **kwargs):
        from .vllm import Vllm
        return Vllm(model_id_or_dir, **kwargs)



