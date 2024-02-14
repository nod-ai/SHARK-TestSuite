import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

test_modelname = "facebook/opt-125m"
quantizedmodelname = "jlsilva/facebook-opt-125m-gptq4bit"
kwargs = {
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}
quantization_config = GPTQConfig(bits=8, disable_exllama=True)
kwargs["quantization_config"] = quantization_config
kwargs["device_map"] = "cpu"
model = AutoModelForCausalLM.from_pretrained(quantizedmodelname, **kwargs)
# model.output_hidden_states = False
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
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
)
print("Input:", test_input)
print("Output:", test_output)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(test_output[0]))
# Do not enforce any particular strategy for getting torch MLIR
# By default set it to None, set it to
# 'compile' : to force using torch_mllir.compile
# 'fximport' : to force using PyTorch 2.0 Fx Import
test_torchmlircompile = None
