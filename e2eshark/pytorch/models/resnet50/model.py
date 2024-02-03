import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from torchvision.models import resnet50, ResNet50_Weights

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

test_input = test_input = torch.randn(1, 3, 224, 224)
test_output = model(test_input)
print("Input:", test_input)
print("Output:", test_output)
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
