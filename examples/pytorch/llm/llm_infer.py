# ### Setting up experimental environment.
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import warnings
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import torch
from swift import LoRAConfig, Swift
from transformers import GenerationConfig, TextStreamer
from utils import (DATASET_MAPPING, DEFAULT_PROMPT, MODEL_MAPPING, get_dataset,
                   get_model_tokenizer, inference, parse_args, process_dataset,
                   tokenize_function)

from modelscope import get_logger

warnings.warn(
    'This directory has been migrated to '
    'https://github.com/modelscope/swift/tree/main/examples/pytorch/llm, '
    'and the files in this directory are no longer maintained.',
    DeprecationWarning)

logger = get_logger()


@dataclass
class InferArguments:
    model_type: str = field(
        default='qwen-7b', metadata={'choices': list(MODEL_MAPPING.keys())})
    sft_type: str = field(
        default='lora', metadata={'choices': ['lora', 'full']})
    ckpt_path: str = '/path/to/your/iter_xxx.pth'
    eval_human: bool = False  # False: eval test_dataset
    ignore_args_error: bool = False  # True: notebook compatibility

    dataset: str = field(
        default='alpaca-en,alpaca-zh',
        metadata={'help': f'dataset choices: {list(DATASET_MAPPING.keys())}'})
    dataset_seed: int = 42
    dataset_sample: int = 20000  # -1: all dataset
    dataset_test_size: float = 0.01
    prompt: str = DEFAULT_PROMPT
    max_length: Optional[int] = 2048

    lora_target_modules: Optional[List[str]] = None
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout_p: float = 0.1

    max_new_tokens: int = 512
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 0.9

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = MODEL_MAPPING[
                self.model_type]['lora_TM']

        if not os.path.isfile(self.ckpt_path):
            raise ValueError(
                f'Please enter a valid ckpt_path: {self.ckpt_path}')


def llm_infer(args: InferArguments) -> None:
    # ### Loading Model and Tokenizer
    support_bf16 = torch.cuda.is_bf16_supported()
    if not support_bf16:
        logger.warning(f'support_bf16: {support_bf16}')

    kwargs = {'low_cpu_mem_usage': True, 'device_map': 'auto'}
    model, tokenizer, _ = get_model_tokenizer(
        args.model_type, torch_dtype=torch.bfloat16, **kwargs)

    # ### Preparing lora
    if args.sft_type == 'lora':
        lora_config = LoRAConfig(
            target_modules=args.lora_target_modules,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout_p,
            pretrained_weights=args.ckpt_path)
        logger.info(f'lora_config: {lora_config}')
        model = Swift.prepare_model(model, lora_config)
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
    elif args.sft_type == 'full':
        state_dict = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f'args.sft_type: {args.sft_type}')

    # ### Inference
    tokenize_func = partial(
        tokenize_function,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_length=args.max_length)
    streamer = TextStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    logger.info(f'generation_config: {generation_config}')

    if args.eval_human:
        while True:
            instruction = input('<<< ')
            data = {'instruction': instruction}
            input_ids = tokenize_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer, generation_config)
            print('-' * 80)
    else:
        dataset = get_dataset(args.dataset.split(','))
        _, test_dataset = process_dataset(dataset, args.dataset_test_size,
                                          args.dataset_sample,
                                          args.dataset_seed)
        mini_test_dataset = test_dataset.select(range(10))
        del dataset
        for data in mini_test_dataset:
            output = data['output']
            data['output'] = None
            input_ids = tokenize_func(data)['input_ids']
            inference(input_ids, model, tokenizer, streamer, generation_config)
            print()
            print(f'[LABELS]{output}')
            print('-' * 80)
            # input('next[ENTER]')


if __name__ == '__main__':
    args, remaining_argv = parse_args(InferArguments)
    if len(remaining_argv) > 0:
        if args.ignore_args_error:
            logger.warning(f'remaining_argv: {remaining_argv}')
        else:
            raise ValueError(f'remaining_argv: {remaining_argv}')
    llm_infer(args)
