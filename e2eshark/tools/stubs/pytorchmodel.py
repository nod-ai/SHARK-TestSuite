# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import sys, argparse
import torch_mlir
import numpy
import io

# Fx importer related
from typing import Optional
import torch.export
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir import fx

# old torch_mlir.compile path
from torch_mlir import torchscript

msg = "The script to run a model test"
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
    "-p",
    "--torchmlircompile",
    choices=["compile", "fximport"],
    default="fximport",
    help="Use torch_mlir.compile, or Fx importer",
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

if not outfileprefix:
    outfileprefix = test_modelname

outfileprefix += "." + dtype

# test_input and test_output are defined in model.py as torch tensor which
# is prepended to this file

if dtype == "bf16":
    model = model.to(torch.bfloat16)
    # casting input to bfloat16 crashes torch.onnx.export, so skip it
    test_input = test_input.to(torch.bfloat16)

if runmode == "onnx" or runmode == "ort":
    onnx_name = outfileprefix + ".onnx"
    onnx_program = torch.onnx.export(model, test_input, onnx_name)
elif runmode == "direct":
    torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
    torch_mlir_model = None
    # override mechanism to get torch MLIR as per model
    if args.torchmlircompile == "compile" or test_torchmlircompile == "compile":
        torch_mlir_model = torchscript.compile(
            model,
            (test_input),
            output_type="torch",
            use_tracing=True,
            verbose=False,
        )
    else:
        torch_mlir_model = fx.export_and_import(model, test_input, model_name=args.outfileprefix)
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())

inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"
torch.save(test_input, inputsavefilename)
torch.save(test_output, outputsavefilename)
