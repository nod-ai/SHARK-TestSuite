import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import OPTModel, AutoTokenizer

test_model_name = "facebook/opt-125M"
model = OPTModel.from_pretrained(
    test_model_name,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(test_model_name)
test_input = torch.tensor([tokenizer.encode("The test prommpt")])
test_output = model(test_input)[0]
print("Input:", test_input)
print("Output:", test_output)
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
