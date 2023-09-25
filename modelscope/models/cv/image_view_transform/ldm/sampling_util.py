import numpy as np
import torch


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions.
    From https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/utils.py"""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f'input has {x.ndim} dims but target_dims is {target_dims}, which is less'
        )
    return x[(..., ) + (None, ) * dims_to_append]


def renorm_thresholding(x0, value):
    # renorm
    pred_max = x0.max()
    pred_min = x0.min()
    pred_x0 = (x0 - pred_min) / (pred_max - pred_min)  # 0 ... 1
    pred_x0 = 2 * pred_x0 - 1.  # -1 ... 1

    s = torch.quantile(
        rearrange(pred_x0, 'b ... -> b (...)').abs(), value, dim=-1)
    s.clamp_(min=1.0)
    s = s.view(-1, *((1, ) * (pred_x0.ndim - 1)))

    # clip by threshold
    # pred_x0 = pred_x0.clamp(-s, s) / s  # needs newer pytorch  # TODO bring back to pure-gpu with min/max

    # temporary hack: numpy on cpu
    pred_x0 = np.clip(pred_x0.cpu().numpy(), -s.cpu().numpy(),
                      s.cpu().numpy()) / s.cpu().numpy()
    pred_x0 = torch.tensor(pred_x0).to(self.model.device)

    # re.renorm
    pred_x0 = (pred_x0 + 1.) / 2.  # 0 ... 1
    pred_x0 = (pred_max - pred_min) * pred_x0 + pred_min  # orig range
    return pred_x0


def norm_thresholding(x0, value):
    s = append_dims(
        x0.pow(2).flatten(1).mean(1).sqrt().clamp(min=value), x0.ndim)
    return x0 * (value / s)


def spatial_norm_thresholding(x0, value):
    # b c h w
    s = x0.pow(2).mean(1, keepdim=True).sqrt().clamp(min=value)
    return x0 * (value / s)
