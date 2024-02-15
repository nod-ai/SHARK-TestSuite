import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
)

test_modelname = "TheBloke/Llama-2-7B-GPTQ"
kwargs = {
    "torch_dtype": torch.float32,
    "trust_remote_code": True,
}
quantization_config = GPTQConfig(bits=8, disable_exllama=True)
kwargs["quantization_config"] = quantization_config
kwargs["device_map"] = "cpu"
model = AutoModelForCausalLM.from_pretrained(
    test_modelname, low_cpu_mem_usage=True, attn_implementation="eager", **kwargs
)
# model.output_hidden_states = False
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
test_input = encoding["input_ids"].cpu()
# Flag to prevent casting of input to a different dtype
keep_input_dtype = False
test_output = model.generate(
    test_input,
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)
# forward_out = model.forward(test_input)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(test_output[0]))
print("Input:", test_input)
print("Output:", test_output)
test_torchmlircompile = None
