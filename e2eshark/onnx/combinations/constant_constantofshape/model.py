# This file constitutes end part of runmodel.py
# tools/stubs/onnxstartmodel.py + model.py in test dir + this file
# makes up the runmodel.py

# input_name and onnx_output will be declared and set
# by middle model.py

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

# TBD: Use iobinding and ortvalue explicitly as per
# https://onnxruntime.ai/docs/api/python/api_summary.html
# call inference session
# test_output = [session.run([], {input_name: test_input})[0]]
test_output = numpy.array([session.run([], {})[0]], dtype=numpy.int64)
print("Input:", test_input)
print("Output:", test_output)
