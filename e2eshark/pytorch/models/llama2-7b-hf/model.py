import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

modelname = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(modelname)
model = LlamaForCausalLM.from_pretrained(
    modelname,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
    torchscript=True,
)
model.to("cpu")
model.eval()
model.output_hidden_states = False
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
test_input = encoding["input_ids"].cpu()
test_output = model.generate(
    test_input,
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)[0]
attention_mask = encoding["attention_mask"]
print("Prompt:", prompt)
print("Response:", tokenizer.decode(test_output))
print("Input:", test_input)
print("Output:", test_output)
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
