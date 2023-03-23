# Copyright (c) 2022 Zhipu.AI
import torch

from .layers import QuantizedColumnParallelLinear, QuantizedRowParallelLinear


def quantize(model, weight_bit_width):
    """Replace fp16 linear with quantized linear"""

    if torch.distributed.get_rank() == 0:
        print(f'> Quantizing model weight to {weight_bit_width} bits')

    for layer in model.transformer.layers:
        layer.attention.query_key_value = QuantizedColumnParallelLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.attention.query_key_value.weight.to(
                torch.cuda.current_device()),
            input_size=layer.attention.query_key_value.input_size,
            output_size=layer.attention.query_key_value.output_size,
            bias=True,
            gather_output=False,
            params_dtype=torch.half,
            name='query_key_value',
            skip_init=True,
            device=layer.attention.query_key_value.weight.device,
        )
        layer.attention.dense = QuantizedRowParallelLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.attention.dense.weight.to(
                torch.cuda.current_device()),
            input_size=layer.attention.dense.input_size,
            output_size=layer.attention.dense.output_size,
            bias=True,
            input_is_parallel=True,
            params_dtype=torch.half,
            name='dense',
            skip_init=True,
            device=layer.attention.dense.weight.device,
        )
        layer.mlp.dense_h_to_4h = QuantizedColumnParallelLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_h_to_4h.weight.to(
                torch.cuda.current_device()),
            input_size=layer.mlp.dense_h_to_4h.input_size,
            output_size=layer.mlp.dense_h_to_4h.output_size,
            bias=True,
            gather_output=False,
            params_dtype=torch.half,
            name='dense_h_to_4h',
            skip_init=True,
            device=layer.mlp.dense_h_to_4h.weight.device,
        )
        layer.mlp.dense_4h_to_h = QuantizedRowParallelLinear(
            weight_bit_width=weight_bit_width,
            weight=layer.mlp.dense_4h_to_h.weight.to(
                torch.cuda.current_device()),
            input_size=layer.mlp.dense_4h_to_h.input_size,
            output_size=layer.mlp.dense_4h_to_h.output_size,
            bias=True,
            input_is_parallel=True,
            params_dtype=torch.half,
            name='dense_h_to_4h',
            skip_init=True,
            device=layer.mlp.dense_4h_to_h.weight.device,
        )

    return model
