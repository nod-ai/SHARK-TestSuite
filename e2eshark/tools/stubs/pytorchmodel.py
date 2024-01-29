# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import sys, argparse
import torch_mlir
import numpy

# Fx importer related
from typing import Optional
import torch.export
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d


def export_and_import(
    f,
    *args,
    fx_importer: Optional[FxImporter] = None,
    constraints: Optional[torch.export.Constraint] = None,
    **kwargs,
):
    context = ir.Context()
    torch_d.register_dialect(context)

    if fx_importer is None:
        fx_importer = FxImporter(context=context)
    prog = torch.export.export(f, args, kwargs, constraints=constraints)
    fx_importer.import_frozen_exported_program(prog)
    return fx_importer.module_op


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
    "-c",
    "--torchmlir",
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
    test_input = test_input.to(torch.bfloat16)

if runmode == "onnx" or runmode == "ort":
    onnx_name = outfileprefix + ".onnx"
    onnx_program = torch.onnx.export(model, test_input, onnx_name)
elif runmode == "direct":
    torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
    torch_mlir_model = None
    if test_torchmlir:
        # override mechanism to get torch MLIR as per model
        args.torchmlir = test_torchmlir
    if args.torchmlir == "compile":
        torch_mlir_model = torch_mlir.compile(
            model,
            (test_input),
            output_type="torch",
            use_tracing=True,
            verbose=False,
        )
    else:
        torch_mlir_model = export_and_import(model, test_input)
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())

if not isinstance(test_input, list):
    # Now save the input and output as numpy array for testing at later states of tool run
    numpy_test_input = test_input.detach().numpy()
    inputsavefilename = outfileprefix + ".input"
    numpy.save(inputsavefilename, numpy_test_input)
else:
    # TODO: need a way to save and restore
    print("TODO: need a way to save and store an input list")

if not isinstance(test_output, list):
    numpy_test_output = test_output.detach().numpy()
    outputsavefilename = outfileprefix + ".output"
    numpy.save(outputsavefilename, numpy_test_output)
else:
    # TODO: need a way to save and restore
    print("TODO: need a way to save and store an output list")
