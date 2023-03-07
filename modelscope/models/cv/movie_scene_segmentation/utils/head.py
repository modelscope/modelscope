# The implementation here is modified based on BaSSL,
# originally Apache 2.0 License and publicly available at https://github.com/kakaobrain/bassl

import torch.nn as nn
import torch.nn.functional as F


class MlpHead(nn.Module):

    def __init__(self, input_dim=2048, hidden_dim=2048, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=True),
        )

    def forward(self, x):
        # x shape: [b t d] where t means the number of views
        x = self.model(x)
        return F.normalize(x, dim=-1)
