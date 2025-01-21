# The implementation here is modified based on DeOldify, originally MIT License and
# publicly available at https://github.com/jantic/DeOldify/blob/master/fastai/callbacks/hooks.py
import functools
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

NormType = Enum('NormType',
                'Batch BatchZero Weight Spectral Group Instance SpectralGN')


def is_listy(x):
    return isinstance(x, (tuple, list))


class Hook():
    'Create a hook on `m` with `hook_func`.'

    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        'Applies `hook_func` to `module`, `input`, `output`.'
        if self.detach:
            input = (o.detach()
                     for o in input) if is_listy(input) else input.detach()
            output = (
                o.detach()
                for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        'Remove the hook from the model.'
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks():
    'Create several hooks on the modules in `ms` with `hook_func`.'

    def __init__(self, ms, hook_func, is_forward=True, detach=True):
        self.hooks = [Hook(m, hook_func, is_forward, detach) for m in ms]

    def __getitem__(self, i):
        return self.hooks[i]

    def __len__(self):
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        'Remove the hooks from the model.'
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(m, i, o):
    return o if isinstance(o, torch.Tensor) else o if is_listy(o) else list(o)


def hook_outputs(modules, detach=True, grad=False):
    'Return `Hooks` that store activations of all `modules` in `self.stored`'
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def one_param(m):
    'Return the first parameter of `m`.'
    return next(m.parameters())


def dummy_batch(m, size=(64, 64)):
    'Create a dummy batch to go through `m` with `size`.'
    ch_in = in_channels(m)
    return one_param(m).new(1, ch_in,
                            *size).requires_grad_(False).uniform_(-1., 1.)


def dummy_eval(m, size=(64, 64)):
    'Pass a `dummy_batch` in evaluation mode in `m` with `size`.'
    return m.eval()(dummy_batch(m, size))


def model_sizes(m, size=(64, 64)):
    'Pass a dummy input through the model `m` to get the various sizes of activations.'
    with hook_outputs(m) as hooks:
        dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


class PrePostInitMeta(type):
    'A metaclass that calls optional `__pre_init__` and `__post_init__` methods'

    def __new__(cls, name, bases, dct):
        x = super().__new__(cls, name, bases, dct)
        old_init = x.__init__

        def _pass(self):
            pass

        @functools.wraps(old_init)
        def _init(self, *args, **kwargs):
            self.__pre_init__()
            old_init(self, *args, **kwargs)
            self.__post_init__()

        x.__init__ = _init
        if not hasattr(x, '__pre_init__'):
            x.__pre_init__ = _pass
        if not hasattr(x, '__post_init__'):
            x.__post_init__ = _pass
        return x


class Module(nn.Module, metaclass=PrePostInitMeta):
    'Same as `nn.Module`, but no need for subclasses to call `super().__init__`'

    def __pre_init__(self):
        super().__init__()

    def __init__(self):
        pass


def children(m):
    'Get children of `m`.'
    return list(m.children())


def num_children(m):
    'Get number of children modules in `m`.'
    return len(children(m))


def children_and_parameters(m: nn.Module):
    'Return the children of `m` and its direct parameters not registered in modules.'
    children = list(m.children())
    children_p = sum([[id(p) for p in c.parameters()] for c in m.children()],
                     [])
    for p in m.parameters():
        if id(p) not in children_p:
            children.append(ParameterModule(p))
    return children


def flatten_model(m):
    if num_children(m):
        mapped = map(flatten_model, children_and_parameters(m))
        return sum(mapped, [])
    else:
        return [m]


def in_channels(m):
    'Return the shape of the first weight layer in `m`.'
    for layer in flatten_model(m):
        if hasattr(layer, 'weight'):
            return layer.weight.shape[1]
    raise Exception('No weight layer')


def relu(inplace: bool = False, leaky: float = None):
    'Return a relu activation, maybe `leaky` and `inplace`.'
    return nn.LeakyReLU(
        inplace=inplace,
        negative_slope=leaky) if leaky is not None else nn.ReLU(
            inplace=inplace)


def conv_layer(ni,
               nf,
               ks=3,
               stride=1,
               padding=None,
               bias=None,
               is_1d=False,
               norm_type=NormType.Batch,
               use_activ=True,
               leaky=None,
               transpose=False,
               init=nn.init.kaiming_normal_,
               self_attention=False):
    'Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers.'
    if padding is None:
        padding = (ks - 1) // 2 if not transpose else 0
    bn = norm_type in (NormType.Batch, NormType.BatchZero)
    if bias is None:
        bias = not bn
    conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
    conv = conv_func(
        ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding)
    if norm_type == NormType.Weight:
        conv = weight_norm(conv)
    elif norm_type == NormType.Spectral:
        conv = spectral_norm(conv)
    layers = [conv]
    if use_activ:
        layers.append(relu(True, leaky=leaky))
    if bn:
        layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
    if self_attention:
        layers.append(SelfAttention(nf))
    return nn.Sequential(*layers)


def res_block(nf,
              dense=False,
              norm_type=NormType.Batch,
              bottle=False,
              **conv_kwargs):
    'Resnet block of `nf` features. `conv_kwargs` are passed to `conv_layer`.'
    norm2 = norm_type
    if not dense and (norm_type == NormType.Batch):
        norm2 = NormType.BatchZero
    nf_inner = nf // 2 if bottle else nf
    return SequentialEx(
        conv_layer(nf, nf_inner, norm_type=norm_type, **conv_kwargs),
        conv_layer(nf_inner, nf, norm_type=norm2, **conv_kwargs),
        MergeLayer(dense))


def conv1d(ni, no, ks=1, stride=1, padding=0, bias=False):
    'Create and initialize a `nn.Conv1d` layer with spectral normalization.'
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()
    return spectral_norm(conv)


class SelfAttention(Module):
    'Self attention layer for nd.'

    def __init__(self, n_channels):
        self.query = conv1d(n_channels, n_channels // 8)
        self.key = conv1d(n_channels, n_channels // 8)
        self.value = conv1d(n_channels, n_channels)
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x):
        'Notation from https://arxiv.org/pdf/1805.08318.pdf'
        size = x.size()
        x = x.view(*size[:2], -1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


def sigmoid_range(x, low, high):
    'Sigmoid function with range `(low, high)`'
    return torch.sigmoid(x) * (high - low) + low


class SigmoidRange(Module):
    'Sigmoid module with range `(low,x_max)`'

    def __init__(self, low, high):
        self.low, self.high = low, high

    def forward(self, x):
        return sigmoid_range(x, self.low, self.high)


class SequentialEx(Module):
    'Like `nn.Sequential`, but with ModuleList semantics, and can access module input'

    def __init__(self, *layers):
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        res = x
        for layer in self.layers:
            res.orig = x
            nres = layer(res)
            res.orig = None
            res = nres
        return res

    def __getitem__(self, i):
        return self.layers[i]

    def append(self, layer):
        return self.layers.append(layer)

    def extend(self, layer):
        return self.layers.extend(layer)

    def insert(self, i, layer):
        return self.layers.insert(i, layer)


class MergeLayer(Module):
    'Merge a shortcut with the result of the module by adding them or concatenating thme if `dense=True`.'

    def __init__(self, dense: bool = False):
        self.dense = dense

    def forward(self, x):
        return torch.cat([x, x.orig], dim=1) if self.dense else (x + x.orig)


class PixelShuffle_ICNR(Module):
    'Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, and `weight_norm`.'

    def __init__(self,
                 ni: int,
                 nf: int = None,
                 scale: int = 2,
                 blur: bool = False,
                 norm_type=NormType.Weight,
                 leaky: float = None):
        nf = ni if nf is None else nf
        self.conv = conv_layer(
            ni, nf * (scale**2), ks=1, norm_type=norm_type, use_activ=False)
        self.shuf = nn.PixelShuffle(scale)
        # Blurring over (h*w) kernel
        # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
        # - https://arxiv.org/abs/1806.02658
        self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
        self.blur = nn.AvgPool2d(2, stride=1)
        self.relu = relu(True, leaky=leaky)

    def forward(self, x):
        x = self.shuf(self.relu(self.conv(x)))
        return self.blur(self.pad(x)) if self.blur else x
