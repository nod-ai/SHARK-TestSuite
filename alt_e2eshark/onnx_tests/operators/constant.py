# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto
from onnx.helper import make_tensor_value_info, make_tensor

from ..helper_classes import BuildAModel
from e2e_testing.registry import register_with_name

@register_with_name("bool_tensor_constant")
class BoolTensorConstant(BuildAModel):
    def construct_i_o_value_info(self):
        self.input_vi = [
            make_tensor_value_info('unused_input_to_make_the_test_framework_happy', TensorProto.INT64, [1]),
        ]
        self.output_vi = [
            make_tensor_value_info('result', TensorProto.BOOL, [2, 2]),
        ]

    def construct_nodes(self):
        self.initializers = [
            make_tensor('result', TensorProto.BOOL, dims=[2, 2], vals=[True, False, False, True]),
        ]
        self.node_list = []
