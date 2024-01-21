import sys, argparse
import torch
import torch.nn as nn
import torch_mlir


class op_conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(8, 10, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        )

    def forward(self, x):
        return self.layers(x)

    def name(self):
        return self.__class__.__name__


model = op_conv2d()
test_input = torch.randn(2, 8, 12, 16)
