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
    "--dtype",
    choices=["fp32", "bf16"],
    default="fp32",
    help="Tensor datatype to use",
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
dtype = args.dtype
runmode = args.mode
outfileprefix = args.outfileprefix
outfileprefix += "." + dtype
inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"

# test_input and test_output are defined in model.py as numpy array which
# is prepended to this file
# to have uniform way to run test
# same input and output as torch .pt
pttest_input = torch.from_numpy(test_input)
pttest_output = torch.from_numpy(test_output)
print(pttest_input, pttest_output)
torch.save(pttest_input, inputsavefilename)
torch.save(pttest_output, outputsavefilename)
