# Copyright 2022-2023 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import torch.nn as nn


class ClassifierHead(nn.Module):
    """The implementation of classification head.

    Attributes:
        dim: An integer indicating the hidden dimension.
        num_classes: A string indicating the number of class.
        dropout_rate: A float indicating the dropout rate.
    """

    def __init__(self, dim, num_classes, dropout_rate=0):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        return self.fc(x)
