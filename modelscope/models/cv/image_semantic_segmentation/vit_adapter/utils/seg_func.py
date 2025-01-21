# The implementation is adopted from VitAdapter,
# made publicly available under the Apache License at https://github.com/czczup/ViT-Adapter.git

import warnings

import torch.nn.functional as F


def seg_resize(input,
               size=None,
               scale_factor=None,
               mode='nearest',
               align_corners=None,
               warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')

    try:
        return F.interpolate(input, size, scale_factor, mode, align_corners)
    except ValueError:
        if isinstance(size, tuple):
            if len(size) == 3:
                size = size[:2]
        return F.interpolate(input, size, scale_factor, mode, align_corners)


def add_prefix(inputs, prefix):
    """Add prefix for dict.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix to add.

    Returns:

        dict: The dict with keys updated with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs
