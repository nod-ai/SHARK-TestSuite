# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import sys, argparse
import torch_mlir
import numpy
import io, pickle

# Fx importer related
from typing import Optional
import torch.export
from torch_mlir.extras.fx_importer import FxImporter
from torch_mlir import ir
from torch_mlir.dialects import torch as torch_d
from torch_mlir import fx

# old torch_mlir.compile path
from torch_mlir import torchscript
from utils import getOutputTensorList

msg = "The script to run a model test"
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
    "-p",
    "--torchmlirimport",
    choices=["compile", "fximport"],
    default="fximport",
    help="Use torch_mlir.torchscript.compile, or Fx importer",
)
parser.add_argument(
    "-o",
    "--outfileprefix",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
runmode = args.mode
outfileprefix = args.outfileprefix + "." + args.todtype


def getTorchDType(dtypestr):
    if dtypestr == "fp32":
        return torch.float32
    elif dtypestr == "fp16":
        return torch.float16
    elif dtypestr == "bf16":
        return torch.bfloat16
    else:
        print("Unknown dtype {dtypestr} returning torch.float32")
        return torch.float32


if args.todtype != "default":
    # convert the model to given dtype
    dtype = getTorchDType(args.todtype)
    model = model.to(dtype)
    # not all model need the input re-casted
    if E2ESHARK_CHECK["inputtodtype"]:
        E2ESHARK_CHECK["input"] = E2ESHARK_CHECK["input"].to(dtype)
    E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])

if runmode == "onnx" or runmode == "ort":
    onnx_name = outfileprefix + ".onnx"
    onnx_program = torch.onnx.export(model, E2ESHARK_CHECK["input"], onnx_name)
elif runmode == "direct":
    torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
    torch_mlir_model = None
    # override mechanism to get torch MLIR as per model
    if (
        args.torchmlirimport == "compile"
        or E2ESHARK_CHECK["torchmlirimport"] == "compile"
    ):
        torch_mlir_model = torchscript.compile(
            model,
            (E2ESHARK_CHECK["input"]),
            output_type="torch",
            use_tracing=True,
            verbose=False,
        )
    else:
        torch_mlir_model = fx.export_and_import(model, E2ESHARK_CHECK["input"])
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())

inputsavefilename = outfileprefix + ".input.pt"
outputsavefilename = outfileprefix + ".goldoutput.pt"

test_input_list = E2ESHARK_CHECK["input"]
test_output_list = E2ESHARK_CHECK["output"]

if not isinstance(E2ESHARK_CHECK["input"], list):
    test_input_list = [E2ESHARK_CHECK["input"]]

if isinstance(test_output_list, tuple):
    # handles only nested tuples for now
    print(f"Found tuple {len(test_output_list)} {test_output_list}")
    test_output_list = getOutputTensorList(E2ESHARK_CHECK["output"])

# model result expected to be List[Tensors]
if not isinstance(test_output_list, list):
    test_output_list = [E2ESHARK_CHECK["output"]]

test_input_list_save = [t.detach() for t in test_input_list]
test_output_list_save = [t.detach() for t in test_output_list]
torch.save(test_input_list_save, inputsavefilename)
torch.save(test_output_list_save, outputsavefilename)

# Save the E2ESHARK_CHECK
with open("E2ESHARK_CHECK.pkl", "wb") as tchkf:
    pickle.dump(E2ESHARK_CHECK, tchkf)
