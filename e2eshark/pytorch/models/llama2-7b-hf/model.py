import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

modelname = "meta-llama/Llama-2-7b-hf"

tokenizer = LlamaTokenizer.from_pretrained(modelname)
model = LlamaForCausalLM.from_pretrained(modelname, low_cpu_mem_usage=True)
model.to("cpu")
prompt = "What is nature of our existence?"
batch = tokenizer(prompt, return_tensors="pt")
test_input = batch["input_ids"].cpu()
test_output = model.generate(
    test_input,
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)[0]
print("Prompt:", prompt)
print("Response:", tokenizer.decode(test_output))
print("Input:", test_input)
print("Output:", test_output)
