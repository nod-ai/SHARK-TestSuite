# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# run.py creates runmodel.py by concatenating this file model.py
# and tools/stubs/onnxmodel.py
# Description: testing ScatterND
# See https://onnx.ai/onnx/intro/python.html for intro on creating
# onnx model using python onnx API
import numpy, torch, sys
import onnxruntime
from onnx import numpy_helper, TensorProto, save_model
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
from onnx.checker import check_model

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# Create an input (ValueInfoProto)
D = make_tensor_value_info("D", TensorProto.FLOAT, [4, 4, 4])
I = make_tensor_value_info("I", TensorProto.INT64, [2, 1])
U = make_tensor_value_info("U", TensorProto.FLOAT, [2, 4, 4])

# Create an output
Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 4, 4])

# Create a node (NodeProto)
scatter_nd_node = make_node(
    "ScatterND", ["D", "I", "U"], ["Z"], "scatter_nd_node"
    #op_type #  # inputs  # outputs #node_name
)

# Create the graph (GraphProto)
graph = make_graph(
    [scatter_nd_node],
    "scatter_nd_graph",
    [D, I, U],
    [Z],
)

# Create the model (ModelProto)
onnx_model = make_model(graph)
onnx_model.opset_import[0].version = 16

# Save the model
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


session = onnxruntime.InferenceSession("model.onnx", None)
model_input_D = numpy.random.randn(4, 4, 4).astype(numpy.float32)
model_input_I = numpy.random.randint(2, size=(2, 1)).astype(numpy.int64)
model_input_U = numpy.random.randn(2, 4, 4).astype(numpy.float32)
inputs = session.get_inputs()
outputs = session.get_outputs()

model_output = session.run(
    [outputs[0].name],
    {inputs[0].name: model_input_D, 
     inputs[1].name: model_input_I,
     inputs[2].name: model_input_U},
)

E2ESHARK_CHECK["input"] = [
    torch.from_numpy(model_input_D),
    torch.from_numpy(model_input_I),
    torch.from_numpy(model_input_U),
]
E2ESHARK_CHECK["output"] = [torch.from_numpy(arr) for arr in model_output]

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
