# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing MLP
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model


# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])

# Weights and Biases
Wx = make_tensor_value_info("Wx", TensorProto.FLOAT, [4, 5])
Bx = make_tensor_value_info("Bx", TensorProto.FLOAT, [3, 5])

# Intermediate
I1 = make_tensor_value_info("I1", TensorProto.FLOAT, [3, 5])
I2 = make_tensor_value_info("I2", TensorProto.FLOAT, [3, 5])

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 5])

# Create a node (NodeProto)
gemmnode = make_node(
    "Gemm", ["X", "Wx"], ["I1"], "gemmnode"  # node name  # inputs  # outputs
)

addnode = make_node(
    "Add", ["I1", "Bx"], ["I2"], "addnode"  # node name  # inputs  # outputs
)

relunode = make_node(
    "Relu", ["I1"], ["Z"], "relunode"  # node name  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [gemmnode, addnode, relunode],
    "mlpgraph",
    [X, Wx, Bx],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
# save_model(onnx_model, "model.onnx")
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
test_input_X = numpy.random.randn(3, 4).astype(numpy.float32)
init_Wx = numpy.random.randn(4, 5).astype(numpy.float32)
init_Bx = numpy.random.randn(3, 5).astype(numpy.float32)

# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

# test_input and test_output are list of numpy arrays
# each index into list is one input or one output in the
# order it appears in the model
test_input = [test_input_X]

test_output = session.run(
    [outputs[0].name],
    {
        inputs[0].name: test_input_X,
        inputs[1].name: init_Wx,
        inputs[2].name: init_Bx,
    },
)

print("Input:", test_input)
print("Output:", test_output)
