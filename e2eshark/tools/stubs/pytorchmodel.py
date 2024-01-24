# This file constitutes end  part of runmodel.py
# this is appended to the model.py in test dir

import sys, argparse
import torch_mlir
import numpy

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
    "-o",
    "--outfileprefix",
    help="Prefix of output files written by this model",
)
args = parser.parse_args()
dtype = args.dtype
runmode = args.mode
outfileprefix = args.outfileprefix

if not outfileprefix:
    outfileprefix = model.name()

outfileprefix += "." + dtype

if dtype == "bf16":
    model = model.to(torch.bfloat16)
    test_input = test_input.to(torch.bfloat16)

if runmode == "onnx" or runmode == "ort":
    onnx_name = outfileprefix + ".onnx"
    onnx_program = torch.onnx.export(model, test_input, onnx_name)
elif runmode == "direct":
    torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
    ts_model = torch.jit.script(model)
    torch_mlir_model = torch_mlir.compile(
        ts_model,
        (test_input),
        output_type="torch",
        use_tracing=True,
        verbose=False,
    )
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())

# Now save the input and output as numpy array for testing at later states of tool run
numpy_test_input = test_input.detach().numpy()
numpy_test_output = test_output.detach().numpy()

if args.dtype == "bf16":
    # Unfortunately, numpy does not support bfloat16, so do the casting dance
    numpy_test_input = numpy.float32(numpy_test_input)
    numpy_test_output = numpy.float32(numpy_test_output)

inputsavefilename = outfileprefix + ".input"
numpy.save(inputsavefilename, numpy_test_input)

outputsavefilename = outfileprefix + ".output"
numpy.save(outputsavefilename, numpy_test_input)
