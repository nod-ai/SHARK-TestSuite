# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API

# Issue: https://github.com/llvm/torch-mlir/issues/2764
# Description: ConstantOfShape whose input (tensor shape) is determined by a Constant node
# This construct is present in OPT and causes the ONNX importer to fail because
# of lack of support for ConstantOfShape with non-initializer input

import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

const = make_node(
    "Constant",
    [],
    ["c_shape"],
    "const",
    value=numpy_helper.from_array(numpy.array([4], dtype=numpy.int64)),
)
cofshape = make_node(
    "ConstantOfShape",
    ["c_shape"],
    ["c_out"],
    "cofshape",
    value=numpy_helper.from_array(numpy.array([1], dtype=numpy.int64)),
)

# the part which does not change
outval = make_tensor_value_info("c_out", TensorProto.INT64, [None])
graph = make_graph([const, cofshape], "constgraph", [], [outval])
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19
check_model(onnx_model)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

# test_input and test_output must be numpy
# bfloat16 is not supported by onnxruntime and numpy
# case to/from fp32 to work around at various stages
# start an onnxrt session
session = onnxruntime.InferenceSession("model.onnx", None)
test_input = numpy.array([])
inputs = session.get_inputs()
# This test has no input
input_name = ""
# Get the name of the input of the model
# input_name = inputs.name

test_output = numpy.array([session.run([], {})[0]], dtype=numpy.int64)
print("Input:", test_input)
print("Output:", test_output)
