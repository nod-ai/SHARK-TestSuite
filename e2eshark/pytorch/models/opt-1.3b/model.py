import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import OPTForCausalLM, AutoTokenizer

# These are pieckled and saved and used by tools/stubs python and run.pl.
# If adding new fields, make sure the field has default value and have updated
# tools/stubs and run.pl to handle the new fields
E2ESHARK_CHECK = {
    # this is input applied to the model
    "input": None,
    # this is output gotten from the model
    "output": None,
    # Controls how to import a graph from PyTorch into MLIR, options are: compile or fximport
    "torchmlirimport": "fximport",
    # By default, the input.to(dtype) is called, set it to False to not do so
    "inputtodtype": True,
    # Apply listed function (tools/stub and run.pl must be able to find definition)
    # on output from target in sequence to post process output and compare the final
    # output,
    # Exmaple: "postprocess": [torch.nn.functional.softmax, torch.topk]
    "postprocess": None,
}

test_modelname = "facebook/opt-1.3b"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
model = OPTForCausalLM.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model.to("cpu")
model.eval()
prompt = "What is nature of our existence?"
encoding = tokenizer(prompt, return_tensors="pt")
E2ESHARK_CHECK["input"] = encoding["input_ids"].cpu()
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
model_response = model.generate(
    E2ESHARK_CHECK["input"],
    do_sample=True,
    top_k=50,
    max_length=100,
    top_p=0.95,
    temperature=1.0,
)
print("Prompt:", prompt)
print("Response:", tokenizer.decode(model_response[0]))
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False
