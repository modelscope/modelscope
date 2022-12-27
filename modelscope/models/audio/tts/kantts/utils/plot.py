# Copyright (c) Alibaba, Inc. and its affiliates.

import matplotlib

matplotlib.use('Agg')
try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError('Please install matplotlib.')

plt.set_loglevel('info')


def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 8))
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
