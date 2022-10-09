# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import shutil

import matplotlib
import matplotlib.pylab as plt
import torch

matplotlib.use('Agg')


class AttrDict(dict):

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def build_env(config, config_name, path):
    t_path = os.path.join(path, config_name)
    if config != t_path:
        os.makedirs(path, exist_ok=True)
        shutil.copyfile(config, os.path.join(path, config_name))


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(
        spectrogram, aspect='auto', origin='lower', interpolation='none')
    plt.colorbar(im, ax=ax)

    fig.canvas.draw()
    plt.close()

    return fig


def plot_alignment(alignment, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Input timestep'
    if info is not None:
        xlabel += '\t' + info
    plt.xlabel(xlabel)
    plt.ylabel('Output timestep')
    fig.canvas.draw()
    plt.close()

    return fig


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    checkpoint_dict = torch.load(filepath, map_location=device)
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    torch.save(obj, filepath)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????.pkl')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class ValueWindow():

    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []


def get_model_size(model):
    param_num = sum([p.numel() for p in model.parameters() if p.requires_grad])
    param_size = param_num * 4 / 1024 / 1024
    return param_size


def get_grad_norm(model):
    total_norm = 0
    params = [
        p for p in model.parameters() if p.grad is not None and p.requires_grad
    ]
    for p in params:
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item()**2
    total_norm = total_norm**0.5
    return total_norm


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean, std)


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size,
                                                       -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
