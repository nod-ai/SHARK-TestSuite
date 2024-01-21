import sys, argparse
import torch
import torch.nn as nn
import torch_mlir


class op_linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 4))

    def forward(self, x):
        return self.layers(x)

    def name(self):
        return self.__class__.__name__


model = op_linear()
test_input = torch.randn(8, 3)
