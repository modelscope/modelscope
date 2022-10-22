# import torch
# from mpu.transformer import GPT2ParallelSelfAttention
#
# b = torch.arange(2) * 1000
# h = torch.arange(3) * 100
# pos_seq = torch.arange(9, -1, -1)
# query = torch.arange(7) * 10
# s = pos_seq.unsqueeze(0) + query.unsqueeze(1)
# s = b.view(-1, 1, 1, 1) + h.view(1, -1, 1, 1) + s
# s = GPT2ParallelSelfAttention._rel_shift(s)
# print(s)

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
    steps = np.arange(0, 400000, 10, dtype=np.long)
    rates = []
    for step in steps:
        lr_scheduler.num_iters = step
        rates.append(lr_scheduler.get_lr())
    print(rates)
    plt.plot(steps, rates)
    plt.savefig('lr.pdf', format='pdf')
