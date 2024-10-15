# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_tensor

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_with_name, register_test

class ConvTransposePads(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3]),
            make_tensor_value_info("W", TensorProto.FLOAT, [1, 2, 5, 5]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 3]),
        ]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            pads=[1, 2, 1, 2],
        )

register_test(ConvTransposePads, "convtranspose_pads_test")

class ConvTransposeAutoPad(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3]),
            make_tensor_value_info("W", TensorProto.FLOAT, [1, 2, 4, 4]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 6, 6]),
        ]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            auto_pad="SAME_UPPER",
            strides=[2, 2],
        )

register_test(ConvTransposeAutoPad, "convtranspose_autopad_test")

class ConvTransposeAutoPadOutputShape(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 3, 3]),
            make_tensor_value_info("W", TensorProto.FLOAT, [1, 2, 3, 3]),
        ]
        self.output_vi = [
            make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2, 5, 5]),
        ]

    def construct_nodes(self):
        app_node = self.get_app_node()
        app_node(
            "ConvTranspose",
            ["X", "W"],
            ["Y"],
            output_shape=[5, 5],
            auto_pad="SAME_UPPER",
            strides=[2, 2],
        )

register_test(ConvTransposeAutoPadOutputShape, "convtranspose_autopad_output_shape_test")
