from dataclasses import dataclass
from typing import Callable

import torch


class Tuner:

    def tune(self, **kwargs):
        pass

    def add_hook(self, trainer):
        pass
