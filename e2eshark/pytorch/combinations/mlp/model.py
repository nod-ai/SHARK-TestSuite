import torch
import torch.nn as nn


class mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # 3 input, 4 output
            nn.Linear(3, 4),
            nn.ReLU(),
            # 3 input, 5 output
            nn.Linear(4, 5),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)

    def name(self):
        return self.__class__.__name__


model = mlp()
test_input = torch.randn(8, 3)
