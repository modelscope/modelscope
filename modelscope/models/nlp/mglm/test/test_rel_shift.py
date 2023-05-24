# Copyright (c) 2022 Zhipu.AI

import matplotlib.pyplot as plt
import numpy as np
from learning_rates import AnnealingLR
from torch.nn.modules import Linear
from torch.optim import Adam


def main():
    model = Linear(10, 10)
    optimizer = Adam(model.parameters())
    lr_scheduler = AnnealingLR(
        optimizer,
        start_lr=0.00015,
        warmup_iter=3000,
        num_iters=300000,
        decay_style='cosine',
        decay_ratio=0.1)
    steps = np.arange(0, 400000, 10, dtype=int)
    rates = []
    for step in steps:
        lr_scheduler.num_iters = step
        rates.append(lr_scheduler.get_lr())
    print(rates)
    plt.plot(steps, rates)
    plt.savefig('lr.pdf', format='pdf')
