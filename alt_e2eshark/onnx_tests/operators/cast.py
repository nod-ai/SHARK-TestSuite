# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_node, make_tensor_value_info, make_attribute

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_test

class CastModel(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info("X", TensorProto.BFLOAT16, [1]),
        ]
        self.output_vi = [make_tensor_value_info("Y", TensorProto.FLOAT16, [1])]

    def construct_nodes(self):
        cast_node = make_node("Cast", ["X"], ["Y"], "castnode")
        cast_node.attribute.append(make_attribute("to", 10))
        self.node_list.append(cast_node)

register_test(CastModel, "cast_test")

