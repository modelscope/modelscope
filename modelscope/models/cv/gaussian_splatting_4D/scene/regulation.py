import abc
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn



def compute_plane_tv(t):
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h-1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w-1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h-1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h-2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def step(self, global_step):
        pass

    def _regularize(self, model, **kwargs):
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
            for grid_id in spatial_grids:
                total += compute_plane_tv(grids[grid_id])
            for grid in grids:
                # grid: [1, c, h, w]
                total += compute_plane_tv(grid)
        return total


class TimeSmoothness(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        return torch.as_tensor(total)



class L1ProposalNetwork(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-proposal-network', initial_value)

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        grids = [p.grids for p in model.proposal_networks]
        total = 0.0
        for pn_grids in grids:
            for grid in pn_grids:
                total += torch.abs(grid).mean()
        return torch.as_tensor(total)


class DepthTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('tv-depth', initial_value)

    def _regularize(self, model, model_out, **kwargs) -> torch.Tensor:
        depth = model_out['depth']
        tv = compute_plane_tv(
            depth.reshape(64, 64)[None, None, :, :]
        )
        return tv


class L1TimePlanes(Regularizer):
    def __init__(self, initial_value, what='field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_value)
        self.what = what

    def _regularize(self, model, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field':
            multi_res_grids = model.field.grids
        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        return torch.as_tensor(total)

