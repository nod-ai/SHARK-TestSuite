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


model = mlp()
test_input = torch.randn(8, 3)
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
