# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey)
# Copyright (c) 2024 Alibaba, Inc. and its affiliates.
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import math
import random
from typing import Optional, Tuple, Union

# import k2
import torch
import torch.nn as nn
from torch import Tensor
from torch.cuda.amp import custom_bwd, custom_fwd


def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


# RuntimeError: Exporting the operator logaddexp to ONNX opset version
# 14 is not supported. Please feel free to request support or submit
# a pull request on PyTorch GitHub.
#
# The following function is to solve the above error when exporting
# models to ONNX via torch.jit.trace()
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    # Caution(fangjun): Put torch.jit.is_scripting() before
    # torch.onnx.is_in_onnx_export();
    # otherwise, it will cause errors for torch.jit.script().
    #
    # torch.logaddexp() works for both torch.jit.script() and
    # torch.jit.trace() but it causes errors for ONNX export.
    #
    if torch.jit.is_scripting():
        # Note: We cannot use torch.jit.is_tracing() here as it also
        # matches torch.onnx.export().
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        # for torch.jit.trace()
        return torch.logaddexp(x, y)


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(
            *[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(
            *[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)]
        )

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise linear
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p cross.
        """
        assert isinstance(p, PiecewiseLinear), type(p)

        # get sorted x-values without repetition.
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return (
            f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"
        )

    def __float__(self):
        batch_count = self.batch_count
        if (
            batch_count is None
            or not self.training
            or torch.jit.is_scripting()
            or torch.jit.is_tracing()
        ):
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(
                    f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}"
                )
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(
                self.schedule + x.schedule, default=self.default + x.default
            )

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(
                self.schedule.max(x.schedule), default=max(self.default, x.default)
            )


FloatLike = Union[float, ScheduledFloat]


class SoftmaxFunction(torch.autograd.Function):
    """
    Tries to handle half-precision derivatives in a randomized way that should
    be more accurate for training than the default behavior.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        # if x dtype is float16, x.softmax() returns a float32 because
        # (presumably) that op does not support float16, and autocast
        # is enabled.
        if torch.is_autocast_enabled():
            ans = ans.to(torch.float16)
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch.cuda.amp.autocast(enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def inplace_softmax(tensor, dim):
    # Subtract the maximum value from each Tensor to prevent overflow.
    max_vals, _ = tensor.max(dim=dim, keepdim=True)
    tensor.sub_(max_vals)

    # # calculate logsumexp
    # log_sum_exp = torch.logsumexp(tensor, dim=dim, keepdim=True)
    #
    # # minus logsumexp
    # tensor.sub_(log_sum_exp)
    #
    # # Compute the exponential of each element, and store the results in-place.
    # tensor.exp_()


    # Compute the exponential of each element, and store the results in-place.
    tensor.exp_()


    # Compute the sum along the specified dimension, and store the result in-place.
    sum_exp = tensor.sum(dim=dim, keepdim=True)

    # Divide each element by the sum along that dimension, and store the result in-place.
    tensor.div_(sum_exp)
    # tensor.add_(1e-8)

    return tensor

def softmax(x: Tensor, dim: int):
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)
        # inplace operator
        # return inplace_softmax(x, dim)

    return SoftmaxFunction.apply(x, dim)


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans




class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    """
    Behaves like a depthwise 1d convolution, except that it is causal in
    a chunkwise way, as if we had a block-triangular attention mask.
    The chunk size is provided at test time (it should probably be
    kept in sync with the attention mask).

    This has a little more than twice the parameters of a conventional
    depthwise conv1d module: we implement it by having one
    depthwise convolution, of half the width, that is causal (via
    right-padding); and one depthwise convolution that is applied only
    within chunks, that we multiply by a scaling factor which depends
    on the position within the chunk.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        half_kernel_size = (kernel_size + 1) // 2
        # will pad manually, on one side.
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )

        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        # first row is correction factors added to the scale near the left edge of the chunk,
        # second row is correction factors added to the scale near the right edge of the chunk,
        # both of these are added to a default scale of 1.0.
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                torch.nn.init.uniform_(
                    self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale
                )

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        """Forward function.

        Args:
               x: a Tensor of shape (batch_size, channels, seq_len)
        chunk_size: the chunk size, in frames; does not have to divide seq_len exactly.
        """
        (batch_size, num_channels, seq_len) = x.shape

        # half_kernel_size = self.kernel_size + 1 // 2
        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(
            batch_size * num_chunks, num_channels, chunk_size
        )
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size)

        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(
            batch_size, num_chunks, num_channels, chunk_size
        ).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[
            ..., :seq_len
        ]

        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int):
        """Returns tensor of shape (num_channels, chunk_size) that will be used to
        scale the output of self.chunkwise_conv."""
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(
                channels, t, device=left_edge.device, dtype=left_edge.dtype
            )
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Streaming Forward function.

        Args:
            x: a Tensor of shape (batch_size, channels, seq_len)
            cache: cached left context of shape (batch_size, channels, left_pad)
        """
        (batch_size, num_channels, seq_len) = x.shape

        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        # Pad cache
        assert cache.shape[-1] == left_pad, (cache.shape[-1], left_pad)
        x = torch.cat([cache, x], dim=2)
        # Update cache
        cache = x[..., -left_pad:]

        x_causal = self.causal_conv(x)
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size=seq_len)
        x_chunk = x_chunk * chunk_scale

        return x_chunk + x_causal, cache


def penalize_abs_values_gt(
    x: Tensor, limit: float, penalty: float, name: str = None
) -> Tensor:
    """
    Returns x unmodified, but in backprop will put a penalty for the excess of
    the absolute values of elements of x over the limit "limit".  E.g. if
    limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.

    The name is for randomly printed debug info.
    """
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    # The following is a memory efficient way to penalize the absolute values of
    # x that's over the limit.  (The memory efficiency comes when you think
    # about which items torch needs to cache for the autograd, and which ones it
    # can throw away).  The numerical value of aux_loss as computed here will
    # actually be larger than it should be, by limit * over_limit.sum(), but it
    # has the same derivative as the real aux_loss which is penalty * (x.abs() -
    # limit).relu().
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    # note: we don't do sum() here on aux)_loss, but it's as if we had done
    # sum() due to how with_loss() works.
    x = with_loss(x, aux_loss, name)
    # you must use x for something, or this will be ineffective.
    return x


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        if random.random() < 0.002 and name is not None:
            loss_sum = y.sum().item()
            logging.info(f"WithLoss: name={name}, loss-sum={loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    # returns x but adds y.sum() to the loss function.
    return WithLoss.apply(x, y, name)




class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)



# Dropout2 is just like normal dropout, except it supports schedules on the dropout rates.
class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)




class SwooshLFunction(torch.autograd.Function):
    """
    swoosh_l(x) =  log(1 + exp(x-4)) - 0.08*x - 0.035
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        coeff = -0.08

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

                if not requires_grad:
                    return y

                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = coeff
                ceil = 1.0 + coeff + 0.005

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                    grad
                )
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.float16)
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.

        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        # if not x.requires_grad:
        #     return k2.swoosh_l_forward(x)
        # else:
        #     return k2.swoosh_l(x)
        return SwooshLFunction.apply(x)


class SwooshLOnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 4.0) - 0.08 * x - 0.035


class SwooshRFunction(torch.autograd.Function):
    """
     swoosh_r(x) =  log(1 + exp(x-1)) - 0.08*x - 0.313261687

    derivatives are between -0.08 and 0.92.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad

        if x.dtype == torch.float16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

                if not requires_grad:
                    return y
                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = -0.08
                ceil = 0.925

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(
                    grad
                )
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.float16)
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.08
        ceil = 0.925
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshR(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
        # if not x.requires_grad:
        #     return k2.swoosh_r_forward(x)
        # else:
        #     return k2.swoosh_r(x)
        return SwooshRFunction.apply(x)


class SwooshROnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 1.0) - 0.08 * x - 0.313261687


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035

def SwooshLForwardAndDeriv(x: Tensor):
    """
    https://k2-fsa.github.io/k2/python_api/api.html#swoosh-l-forward-and-deriv
    :param x:
    :return:
    """
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)

    deriv = 0.92 - 1 / (1 + x_offset.exp())

    return log_sum - 0.08 * x - 0.035, deriv

# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687

def SwooshRForwardAndDeriv(x: Tensor):
    """
    https://k2-fsa.github.io/k2/python_api/api.html#swoosh-r-forward-and-deriv
    :param x:
    :return:
    """
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)

    deriv = 0.92 - 1 / (1 + x_offset.exp())

    return log_sum - 0.08 * x - 0.313261687, deriv




class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        l = ScaledLinear(
            in_channels, out_channels, bias=bias, initial_scale=initial_scale
        )

        self.weight = l.weight
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", l.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        # if torch.jit.is_scripting() or torch.jit.is_tracing():
        if torch.jit.is_scripting() or torch.jit.is_tracing() or (not self.training):
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
                # x = k2.swoosh_l_forward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
                # x = k2.swoosh_r_forward(x)
            else:
                assert False, self.activation
            return torch.nn.functional.linear(x, self.weight, self.bias)

        # print(f"dropout_p:{float(self.dropout_p)}")
        # print(f"dropout_shared_dim:{self.dropout_shared_dim}")
        # return ActivationDropoutAndLinearFunction.apply(
        #     x,
        #     self.weight,
        #     self.bias,
        #     self.activation,
        #     float(self.dropout_p),
        #     self.dropout_shared_dim,
        # )


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    """

    :param x: (b, c, t, f)
    :param num_channels:
    :return: x: (b, num_channels, t, f)
    """
    if num_channels <= x.shape[1]:
        return x[:, :num_channels, :, :]
    else:
        shape = list(x.shape)
        shape[1] = num_channels - shape[1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=1)





if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

