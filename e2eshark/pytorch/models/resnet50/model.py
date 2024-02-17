import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from torchvision.models import resnet50, ResNet50_Weights

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

test_modelname = "resnet50"
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

E2ESHARK_CHECK["input"] = E2ESHARK_CHECK["input"] = torch.randn(1, 3, 224, 224)
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
