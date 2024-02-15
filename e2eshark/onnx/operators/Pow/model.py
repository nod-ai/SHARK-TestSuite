import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor

# Create an input (ValueInfoProto) for X
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])

# Create an output (ValueInfoProto)
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])  # Same shape as input

# Define a tensor for Y, the exponent, as a constant. For example, raising each element in X to the power of 2.
Y_value = numpy.array(2, dtype=numpy.float32)
Y_tensor = make_tensor("Y", TensorProto.FLOAT, dims=[], vals=Y_value.flatten().astype(float))

# Create a 'Pow' node (NodeProto)
pow_node = make_node(
    "Pow",  # op_type
    ["X", "Y"],  # inputs
    ["Z"],       # outputs
    name="pow_node"
)

# Create the graph (GraphProto)
graph = make_graph(
    [pow_node],  # Nodes in the graph
    "pow_graph",  # Name of the graph
    [X],          # Inputs to the graph
    [Z],          # Outputs of the graph
    initializer=[Y_tensor],  # Initializer for constant tensors
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
inputs = session.get_inputs()
outputs = session.get_outputs()
test_input = [test_input_X] # this line is important: it is used in the test harness code
test_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

print("Input shape:", test_input_X.shape)
print("Output shape:", numpy.array(test_output[0]).shape)
