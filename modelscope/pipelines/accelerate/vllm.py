from typing import List

from modelscope.pipelines.accelerate.base import InferFramework
from modelscope.utils.import_utils import is_vllm_available


class Vllm(InferFramework):

    def __init__(self,
                 model_id_or_dir: str,
                 dtype: str = None,
                 quantization:str = None,
                 tensor_parallel_size: int = None):
        super().__init__(model_id_or_dir)
        if not is_vllm_available():
            raise ImportError(f'Install vllm by `pip install vllm` before using vllm to accelerate inference')

        from vllm import LLM
        if Vllm.check_gpu_compatibility(8) and (dtype is None or dtype == 'bfloat16'):
            dtype = 'float16'
        self.model = LLM(self.model_dir, dtype=dtype,
                         quantization=quantization, tensor_parallel_size=tensor_parallel_size)

    def __call__(self, prompts: List[str], **kwargs) -> List[str]:
        from vllm import SamplingParams
        sampling_params = SamplingParams(**kwargs)
        return [output.outputs[0].text for output in self.model.generate(prompts, sampling_params)]
