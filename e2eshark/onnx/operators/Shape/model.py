import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])  # Example input shape

# The output of 'Shape' is a 1D tensor with length equal to the rank of the input tensor
# Since the input tensor rank is 2 (3, 4), the output shape will be [2]
Y = make_tensor_value_info("Y", TensorProto.INT64, [2])  # Shape's output is always INT64

# Create a 'Shape' node (NodeProto)
shape_node = make_node(
    "Shape",   # op_type
    ["X"],     # inputs
    ["Y"],     # outputs
    "shape_node"  # node name
)

# Create the graph (GraphProto)
graph = make_graph(
    [shape_node],   # Nodes in the graph
    "shape_graph",  # Name of the graph
    [X],            # Inputs to the graph
    [Y],            # Outputs of the graph
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 14  # Set the opset version to ensure compatibility

# Save the model
model_path = "model.onnx"
with open(model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())

# Initialize the ONNX runtime session and run inference
session = onnxruntime.InferenceSession(model_path, None)
test_input_X = numpy.random.randn(3, 4).astype(numpy.float32)  # Match the input shape
test_input = [test_input_X]  # Important for test harness code
inputs = session.get_inputs()
outputs = session.get_outputs()

test_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

# Adjusting test_inputs and test_outputs for validation in test harness
test_inputs = [test_input_X]
test_outputs = [test_output[0]]

print("Input shape:", test_input_X.shape)
print("Output shape:", numpy.array(test_output[0]).shape)
