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


model = op_conv2d()
test_input = torch.randn(2, 8, 12, 16)
# Flag to prevent casting of input to a different dtype
keep_input_dtype = False
test_output = model(test_input)
print("Input:", test_input)
print("Output:", test_output)
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
