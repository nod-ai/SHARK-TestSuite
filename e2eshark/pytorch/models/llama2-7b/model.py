import sys, argparse
import torch
import torch.nn as nn
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer

modelname = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(modelname)
tokenizer = LlamaTokenizer.from_pretrained(modelname)
test_input = "How to make a carrot halwa?"
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto",
)
print("Not redy yet, hangs. exiting...")
sys.exit(1)
sequences = pipeline(
    test_input,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=400,
)

# test_output = []
# for seq in sequences:
#     sentence = seq["generated_text"]
#     print(f"{sentence}")
#     test_output += sentence

# print("Input:", test_input)
# print("Output:", test_output)
