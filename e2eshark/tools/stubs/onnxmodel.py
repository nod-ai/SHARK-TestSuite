# This file constitutes end part of runmodel.py
# this is appended to the model.py in test dir

import numpy
import onnxruntime
import sys, argparse


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
inputsavefilename = outfileprefix + ".input"
outputsavefilename = outfileprefix + ".output"

# test_input and test_output are defined in model.py which
# is prepended to this file
# numpy does not support bfloat16, cast to fp32 and restore back to
# bfloat16 in run.py when gotten value from an inference run that supports
# bfloat16
if args.dtype == "bf16":
    test_input = numpy.float32(test_input)
    test_output = numpy.float32(test_output)

numpy.save(inputsavefilename, test_input)
numpy.save(outputsavefilename, test_output)
