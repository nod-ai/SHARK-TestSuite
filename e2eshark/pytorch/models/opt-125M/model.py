import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import OPTModel, AutoTokenizer

test_model_name = "facebook/opt-125M"


class model_opt_125M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = OPTModel.from_pretrained(
            test_model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]

    def name(self):
        return self.__class__.__name__


model = model_opt_125M()
tokenizer = AutoTokenizer.from_pretrained(test_model_name)
test_input = torch.tensor([tokenizer.encode("The test prommpt")])
test_output = model(test_input)
print("Input:", test_input)
print("Onput:", test_output)
