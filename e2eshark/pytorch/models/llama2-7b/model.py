import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import LlamaForCausalLM, LlamaTokenizer

test_model_name = "meta-llama/Llama-2-7b-hf"


class model_llama2_7b_hf(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = LlamaForCausalLM.from_pretrained(test_model_name)
        self.model.eval()
        self.model.eval()

    def forward(self, tokens):
        attention_mask = torch.ones(tokens.shape, dtype=torch.long)
        return self.model.forward(input_ids=tokens, attention_mask=attention_mask)

    def name(self):
        return self.__class__.__name__


model = model_llama2_7b_hf()
tokenizer = LlamaTokenizer.from_pretrained(test_model_name)
test_input = tokenizer.encode("The llama goes to graze grass", return_tensors="pt")
