# This file constitutes end part of runmodel.py
# this is appended to the model.py in test dir

import numpy
import onnxruntime
import sys, argparse
import torch

msg = "The script to run an ONNX model test"
parser = argparse.ArgumentParser(description=msg, epilog="")

parser.add_argument(
    "-d",
    "--todtype",
    choices=["default", "fp32", "fp16", "bf16"],
    default="default",
    help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
)
parser.add_argument(
    "-m",
    "--mode",
    choices=["direct", "onnx", "ort"],
    default="direct",
    help="Generate torch MLIR, ONNX or ONNX plus ONNX RT stub",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
if args.todtype != "default":
    print(
        "Onnx does not support model.to(dtype). Default dtype of the model will be used."
    )

runmode = args.mode
outfileprefix = args.outfileprefix
outfileprefix += "." + args.todtype
inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"

# test_input and test_output are defined in model.py as
# list of numpy array, each input and output is an element
# of the list.
# same input and output as torch .pt
pttest_input = [torch.from_numpy(arr) for arr in test_input]
pttest_output = [torch.from_numpy(arr) for arr in test_output]
print(pttest_input, pttest_output)
torch.save(pttest_input, inputsavefilename)
torch.save(pttest_output, outputsavefilename)
