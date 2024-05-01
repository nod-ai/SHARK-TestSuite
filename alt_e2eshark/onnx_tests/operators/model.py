import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info
import os

from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
# add a test by setting test_dict["my_test_name"] = OnnxModelInfo(info about test)
class AddModel(OnnxModelInfo):
    def construct_model(self):
        # Create an input (ValueInfoProto)
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [4, 5])

        # Create an output
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [4, 5])

        # Create a node (NodeProto)
        addnode = make_node(
            "Add", ["X", "Y"], ["Z"], "addnode"  # node name  # inputs  # outputs
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            [addnode],
            "main",
            [X, Y],
            [Z],
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(AddModel, "add_test")

class ConcatModel(OnnxModelInfo):
    def construct_model(self):
        # Create an input (ValueInfoProto)
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1, 4, 5])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [2, 4, 5])

        # Create an output
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [3, 4, 5])

        # Create a node (NodeProto)
        concat_node = make_node(
            op_type="Concat",
            inputs=["X", "Y"],
            outputs=["Z"],
            axis=0,
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            nodes=[concat_node],
            name="main",
            inputs=[X, Y],
            outputs=[Z],
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(ConcatModel, "concat_test")


class DynamicQuantizeLinearModel(OnnxModelInfo):
    def construct_model(self):
        input_X = make_tensor_value_info("X", TensorProto.FLOAT, [3, 4]) 
        output_Z = make_tensor_value_info("Z", TensorProto.UINT8, [3, 4])
        output_S = make_tensor_value_info("scale", TensorProto.FLOAT, [])
        output_P = make_tensor_value_info("zp", TensorProto.UINT8, [])

        # Create a 'DQL' node (NodeProto)
        DQL_node = make_node(
            "DynamicQuantizeLinear", ["X"], ["Z", "scale", "zp"], "DQL_node"  # op_type  # inputs  # outputs  # node name
        )

        # Create the graph (GraphProto)
        graph = make_graph(
            [DQL_node],  # Nodes in the graph
            "main",  # Name of the graph
            [input_X],  # Inputs to the graph
            [output_Z, output_S, output_P],  # Outputs of the graph
        )

        # Create the model (ModelProto)
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 11  # Set the opset version to ensure compatibility

        # Save the model
        onnx.save(onnx_model, self.model)

register_test(DynamicQuantizeLinearModel, "dynamic_quanitze_linear_test")