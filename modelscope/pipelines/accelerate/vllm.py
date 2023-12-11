from typing import List, Union

from modelscope.pipelines.accelerate.base import InferFramework
from modelscope.utils.import_utils import is_vllm_available


class Vllm(InferFramework):

    def __init__(self,
                 model_id_or_dir: str,
                 dtype: str = 'auto',
                 quantization: str = None,
                 tensor_parallel_size: int = 1):
        """
        Args:
            dtype: The dtype to use, support `auto`, `float16`, `bfloat16`, `float32`
            quantization: The quantization bit, default None means do not do any quantization.
            tensor_parallel_size: The tensor parallel size.
        """
        super().__init__(model_id_or_dir)
        if not is_vllm_available():
            raise ImportError(
                'Install vllm by `pip install vllm` before using vllm to accelerate inference'
            )

        from vllm import LLM
        if not Vllm.check_gpu_compatibility(8) and (dtype
                                                    in ('bfloat16', 'auto')):
            dtype = 'float16'
        self.model = LLM(
            self.model_dir,
            dtype=dtype,
            quantization=quantization,
            trust_remote_code=True,
            tensor_parallel_size=tensor_parallel_size)

    def __call__(self, prompts: Union[List[str], List[List[int]]],
                 **kwargs) -> List[str]:
        """Generate tokens.
        Args:
            prompts(`Union[List[str], List[List[int]]]`):
                The string batch or the token list batch to input to the model.
            kwargs: Sampling parameters.
        """

        # convert hf generate config to vllm
        do_sample = kwargs.pop('do_sample', None)
        num_beam = kwargs.pop('num_beam', 1)
        max_length = kwargs.pop('max_length', None)
        max_new_tokens = kwargs.pop('max_new_tokens', None)

        # for vllm, default to do_sample/greedy(depends on temperature).
        # for hf, do_sample=false, num_beam=1 -> greedy(default)
        #         do_sample=ture, num_beam=1 -> sample
        #         do_sample=false, num_beam>1 -> beam_search
        if not do_sample and num_beam > 1:
            kwargs['use_beam_search'] = True
        if max_length:
            kwargs['max_tokens'] = max_length - len(prompts[0])
        if max_new_tokens:
            kwargs['max_tokens'] = max_new_tokens

        from vllm import SamplingParams
        sampling_params = SamplingParams(**kwargs)
        if isinstance(prompts[0], str):
            return [
                output.outputs[0].text for output in self.model.generate(
                    prompts, sampling_params=sampling_params)
            ]
        else:
            return [
                output.outputs[0].text for output in self.model.generate(
                    prompt_token_ids=prompts, sampling_params=sampling_params)
            ]

    def model_type_supported(self, model_type: str):
        return any([
            model in model_type.lower() for model in [
                'llama',
                'baichuan',
                'internlm',
                'mistral',
                'aquila',
                'bloom',
                'falcon',
                'gpt',
                'mpt',
                'opt',
                'qwen',
                'aquila',
            ]
        ])
