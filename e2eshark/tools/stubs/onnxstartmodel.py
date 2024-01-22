import numpy as np
import onnxruntime
import sys, argparse

# This file constitutes beginning part of runmodel.py
# this + model.py in test dir + tools/stubs/onnxendmodel.py
# makes up the runmodel.py

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
parser.add_argument(
    "-n",
    "--onnxfilename",
    help="Full path of the onnx model file",
)
args = parser.parse_args()
dtype = args.dtype
runmode = args.mode
outfileprefix = args.outfileprefix
outfileprefix += "." + dtype
onnxfilename = args.onnxfilename
inputsavefilename = outfileprefix + ".input"
outputsavefilename = outfileprefix + ".output"

# Start ONNX runtime session
session = onnxruntime.InferenceSession(onnxfilename, None)
