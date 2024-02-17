import numpy
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info

# Create an input (ValueInfoProto)
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])  # Example input shape

# Create an output (ValueInfoProto)
# The output shape is the same as the input shape for Sigmoid operation
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])  # Same shape as input

# Create a 'Sigmoid' node (NodeProto)
sigmoid_node = make_node(
    "Sigmoid",  # op_type
    ["X"],       # inputs
    ["Z"],       # outputs
    "sigmoid_node"  # node name
)

# Create the graph (GraphProto)
graph = make_graph(
    [sigmoid_node],  # Nodes in the graph
    "sigmoid_graph",  # Name of the graph
    [X],              # Inputs to the graph
    [Z],              # Outputs of the graph
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
test_input = [test_input_X] # this line is important: it is used in the test harness code
inputs = session.get_inputs()
outputs = session.get_outputs()

test_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

print("Input shape:", test_input_X.shape)
print("Output shape:", numpy.array(test_output[0]).shape)
# things that need to be kept constant for every test:
# 1. Define Input and Output Shapes Clearly: Each test must start by defining the input tensor shapes using make_tensor_value_info, ensuring the dimensions match the expected input for the operation being tested.
# 2. There must be a test_inputs and test_outputs list so that the test harness code can access the input and output data for validation.
# 3. The model must be saved as model.onnx in the current working directory.