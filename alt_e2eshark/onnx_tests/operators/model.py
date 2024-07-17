# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import numpy, torch, sys
import onnxruntime
from onnx import TensorProto
import onnx
from onnx.helper import make_model, make_node, make_graph, make_tensor_value_info, make_tensor
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

class PadModel(OnnxModelInfo):
    def construct_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])
        Y = make_tensor_value_info("Y", TensorProto.INT32, [4, 5])
        Z = make_tensor_value_info("Z", TensorProto.FLOAT, [1, 1, 4, 5])
        W = make_tensor_value_info("W", TensorProto.FLOAT, [3, 4, 5])
        XP = make_tensor_value_info("XP", TensorProto.INT64, [4])
        XV = make_tensor_value_info("XV", TensorProto.FLOAT, [])
        YV = make_tensor_value_info("YV", TensorProto.INT32, [])
        ZP = make_tensor_value_info("ZP", TensorProto.INT64, [8])

        XO = make_tensor_value_info("XO", TensorProto.FLOAT, [-1, -1])
        YO = make_tensor_value_info("YO", TensorProto.INT32, [-1, -1])
        ZO = make_tensor_value_info("ZO", TensorProto.FLOAT, [-1, -1, -1, -1])
        # constantNode = make_node(op_type="Constant", inputs=[], outputs=["XP"], value_ints=[2,1,3,4])
        # padnodeX = make_node("Pad", ["X","XP","XV"], ["XO"])
        # padnodeY = make_node(op_type="Pad", inputs=["Y","XP","YV"], outputs=["YO"])
        constantNodeZ = make_node(op_type="Constant", inputs=[], outputs=["ZP"], value_ints=[0,0,2,1,0,0,4,3])
        padnodeZ = make_node(op_type="Pad", inputs=["Z","ZP"], outputs=["ZO"], mode="edge")
        # padgraph = make_graph([constantNode, padnodeX], "main", [X, XV], [XO])
        padgraph = make_graph([constantNodeZ, padnodeZ], "main", [Z], [ZO])
        onnx_model = make_model(padgraph)
        onnx_model.opset_import[0].version = 19

        onnx.save(onnx_model, self.model)

register_test(PadModel, "pad_test")

class ResizeModel(OnnxModelInfo):
    def construct_model(self):
        X = make_tensor_value_info("X", TensorProto.FLOAT, [1,21,65,65])
        C0 = make_tensor_value_info("C0", TensorProto.FLOAT, [4])
        C1 = make_tensor_value_info("C1", TensorProto.FLOAT, [8])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [1, 21, 513, 513])
        C0T = make_tensor("C0T",TensorProto.FLOAT, [4], [1.000000e+00, 1.000000e+00, 7.89230776, 7.89230776])
        C1T = make_tensor("C1T",TensorProto.FLOAT, [8], [0.000000e+00, 0.000000e+00, 0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00, 1.000000e+00])

        const_node0 = make_node(
            op_type="Constant",
            inputs=[],
            outputs=["C0"],
            value=C0T,
        )

        const_node1 = make_node(
            op_type="Constant",
            inputs=[],
            outputs=["C1"],
            value=C1T,
        )

        resize_node = make_node(
            op_type="Resize",
            inputs=["X","C1","C0"],
            outputs=["Y"],
            mode="linear",
            exclude_outside=0,
            coordinate_transformation_mode="half_pixel",
            cubic_coeff_a=-0.75,
            extrapolation_value=0.0,
            nearest_mode="round_prefer_floor",
        )

        graph = make_graph([const_node0, const_node1, resize_node],"main",[X],[Y])
        onnx_model = make_model(graph)
        onnx_model.opset_import[0].version = 19
        onnx.save(onnx_model, self.model)

register_test(ResizeModel, "resize_test")


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
            "DynamicQuantizeLinear",
            ["X"],
            ["Z", "scale", "zp"],
            "DQL_node",  # op_type  # inputs  # outputs  # node name
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
        onnx_model.opset_import[0].version = (
            11  # Set the opset version to ensure compatibility
        )

        # Save the model
        onnx.save(onnx_model, self.model)


register_test(DynamicQuantizeLinearModel, "dynamic_quanitze_linear_test")
