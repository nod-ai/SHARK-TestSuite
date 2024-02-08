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
from onnx import numpy_helper, TensorProto, save_model
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

session = onnxruntime.InferenceSession("model.onnx", None)
test_input_X = numpy.array([])
# There is no input for this one
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

# test_input and test_output are list of numpy arrays
# each index into list is one input or one output in the
# order it appears in the model
test_input = [test_input_X]

test_output = session.run(
    [outputs[0].name],
    {},
)

print("Input:", test_input)
print("Output:", test_output)
