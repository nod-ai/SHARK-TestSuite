import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import (
    make_model,
    make_node,
    make_graph,
    make_tensor_value_info,
    make_tensor,
)

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = E2ESHARK_CHECK_DEF


# Create an input (ValueInfoProto) for X
X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4])

# Create an output (ValueInfoProto)
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4])  # Same shape as input

# Define a tensor for Y, the exponent, as a constant. For example, raising each element in X to the power of 2.
Y_value = numpy.array(2, dtype=numpy.float32)
Y_tensor = make_tensor(
    "Y", TensorProto.FLOAT, dims=[], vals=Y_value.flatten().astype(float)
)

# Create a 'Pow' node (NodeProto)
pow_node = make_node(
    "Pow", ["X", "Y"], ["Z"], name="pow_node"  # op_type  # inputs  # outputs
)

# Create the graph (GraphProto)
graph = make_graph(
    [pow_node],  # Nodes in the graph
    "pow_graph",  # Name of the graph
    [X],  # Inputs to the graph
    [Z],  # Outputs of the graph
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

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

print("Input shape:", test_input_X.shape)
print("Output shape:", numpy.array(model_output[0]).shape)

# Moving to torch to handle bfloat16 as numpy does not support bfloat16
test_input = [torch.from_numpy(test_input_X)]
test_output = [torch.from_numpy(arr) for arr in model_output]

print("Input:", test_input)
print("Output:", test_output)
