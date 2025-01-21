# Copyright (c) 2022 Zhipu.AI

import argparse
import time

import torch
from SwissArmyTransformer import get_args, get_tokenizer
from SwissArmyTransformer.arguments import initialize_distributed
from SwissArmyTransformer.model import GLM130B
from SwissArmyTransformer.mpu import (get_model_parallel_group,
                                      get_model_parallel_rank,
                                      get_model_parallel_world_size)
from SwissArmyTransformer.training import load_checkpoint

from .quantization import quantize


def add_bminf_args(parser):
    """Arguments for BMInf"""
    group = parser.add_argument_group('BMInf')

    group.add_argument(
        '--bminf',
        action='store_true',
        help='Use BMInf to support low resource evaluation')
    group.add_argument(
        '--bminf-memory-limit',
        type=int,
        default=20,
        help='Max memory for model per GPU (in GB)')
    return parser


def add_quantization_args(parser):
    group = parser.add_argument_group('Quantization')

    group.add_argument('--quantization-bit-width', type=int, default=4)
    group.add_argument(
        '--from-quantized-checkpoint',
        type=bool,
        default=True,
        help='Loading from a quantized checkpoint')


def add_initialization_args(parser):
    group = parser.add_argument_group('Initialization')

    group.add_argument(
        '--sequential-initialization',
        action='store_true',
        help=
        'Initialize sequentially in tensor parallel group (reduce CPU RAM for initialization)',
    )


def set_up_model_args(args):
    args.model_parallel_size = 4
    args.num_layers = 70
    args.hidden_size = 12288
    args.inner_hidden_size = 32768
    args.vocab_size = 150528
    args.num_attention_heads = 96
    args.max_sequence_length = 2048
    args.tokenizer_type = 'icetk-glm-130B'
    args.layernorm_order = 'post'
    args.skip_init = True
    args.fp16 = True
    args.mode = 'inference'
    return args


def initialize(extra_args_provider):
    parser = argparse.ArgumentParser(add_help=False)
    add_bminf_args(parser)
    add_quantization_args(parser)
    add_initialization_args(parser)
    GLM130B.add_model_specific_args(parser)
    extra_args_provider(parser)
    known, args_list = parser.parse_known_args()
    args_list += ['--model-parallel-size', '4', '--mode', 'inference']
    args = get_args(args_list)
    args = set_up_model_args(args)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.do_train = False
    initialize_distributed(args)
    return args


def initialize_model_and_tokenizer(args):
    tokenizer = get_tokenizer(args)

    torch.distributed.barrier()
    start = time.time()

    for i in range(get_model_parallel_world_size()):
        if get_model_parallel_rank() == i:
            # Initialize model
            model = GLM130B(args).half()

            if args.from_quantized_checkpoint:
                assert args.quantization_bit_width is not None
                # Quantize model before moving to GPU
                model = quantize(model, args.quantization_bit_width)

            # Load checkpoint
            load_checkpoint(model, args)

            if args.quantization_bit_width is not None and not args.from_quantized_checkpoint:
                # Quantize model before moving to GPU
                model = quantize(model, args.quantization_bit_width)

            if args.bminf:
                import bminf

                if torch.distributed.get_rank() == 0:
                    print(
                        f'> BMInf activated, memory limit: {args.bminf_memory_limit} GB'
                    )
                with torch.cuda.device(args.device):
                    model = bminf.wrapper(
                        model,
                        quantization=False,
                        memory_limit=args.bminf_memory_limit << 30)
            else:
                model = model.to(args.device)
        if args.sequential_initialization:
            torch.distributed.barrier(group=get_model_parallel_group())

    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(f'> Model initialized in {time.time() - start:.1f}s')

    torch.cuda.empty_cache()
    model.eval()

    # generate rotary embedding cache
    original_parallel_output = model.transformer.parallel_output
    model.transformer.parallel_output = True
    with torch.no_grad():
        _, *_ = model(
            torch.ones(
                1,
                args.max_sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.int64),
            torch.arange(
                args.max_sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.int64).view(1, -1),
            torch.randn(
                1,
                1,
                args.max_sequence_length,
                args.max_sequence_length,
                device=torch.cuda.current_device(),
            ) < 0.5,
        )
    model.transformer.parallel_output = original_parallel_output
    torch.distributed.barrier()

    return model, tokenizer
