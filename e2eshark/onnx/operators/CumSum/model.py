# run.pl creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing CumSum
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor
from onnx.checker import check_model

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])

axis_tensor = make_tensor(
    name="axis",
    data_type=TensorProto.INT32,
    dims=(1,),
    vals=[1],
)

axis_node = make_node(
    "Constant",
    inputs=[],
    outputs=["axis"],
    value=axis_tensor,
)

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])

# Create a node (NodeProto)
cumsum_node = make_node(
    "CumSum", ["X", "axis"], ["Z"], "cumsum_node"  # node name  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [axis_node, cumsum_node],  # 'axis_node' is now before 'cumsum_node'
    "cumsum_graph",
    [X],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 19

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

session = onnxruntime.InferenceSession("model.onnx", None)
test_input_X = numpy.random.randn(4, 5).astype(numpy.float32)
inputs = session.get_inputs()
outputs = session.get_outputs()

test_input = [test_input_X]

test_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

print("Input:", test_input)
print("Output:", test_output)