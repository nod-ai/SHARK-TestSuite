import sys, argparse
import torch
import torch.nn as nn
import torch_mlir

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


class op_conv2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(8, 10, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        )

    def forward(self, x):
        return self.layers(x)


model = op_conv2d()
E2ESHARK_CHECK["input"] = torch.randn(2, 8, 12, 16)
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
