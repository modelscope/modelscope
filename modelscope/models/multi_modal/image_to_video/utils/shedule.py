# Copyright (c) Alibaba, Inc. and its affiliates.

import math

import torch


def fn(u):
    return math.cos((u + 0.008) / 1.008 * math.pi / 2)**2


def beta_schedule(schedule,
                  num_timesteps=1000,
                  init_beta=None,
                  last_beta=None):
    '''
    This code defines a function beta_schedule that generates a sequence of beta values based on the given input
    parameters. These beta values can be used in video diffusion processes. The function has the following parameters:
        schedule(str): Determines the type of beta schedule to be generated. It can be 'linear', 'linear_sd',
                      'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value. If not provided, a default value is used based on the
                                    chosen schedule.
        last_beta(float, optional): The final beta value. If not provided, a default value is used based on the
                                    chosen schedule.
    The function returns a PyTorch tensor containing the generated beta values.
    The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values betweeninit_betaandlast_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and the square root
                     oflast_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values forinit_betaandlast_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function, ensuring the values are between 0
                  and 0.999.
    If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    '''
    if schedule == 'linear':
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return torch.linspace(
            init_beta, last_beta, num_timesteps, dtype=torch.float64)
    elif schedule == 'linear_sd':
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
    elif schedule == 'quadratic':
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return torch.linspace(
            init_beta**0.5, last_beta**0.5, num_timesteps,
            dtype=torch.float64)**2
    elif schedule == 'cosine':
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return torch.tensor(betas, dtype=torch.float64)
    else:
        raise ValueError(f'Unsupported schedule: {schedule}')
