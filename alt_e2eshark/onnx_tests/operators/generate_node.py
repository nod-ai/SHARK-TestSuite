# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
from onnx import TensorProto, numpy_helper
import os

from e2e_testing.storage import TestTensors
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
from site import getsitepackages


def get_tensor_from_pb(inputpb):
    proto = TensorProto()
    with open(inputpb, "rb") as f:
        proto.ParseFromString(f.read())
    t = numpy_helper.to_array(proto)
    return TestTensors((t,))


onnx_node_tests_dir = getsitepackages()[0] + "/onnx/backend/test/data/node/"

names = os.listdir(onnx_node_tests_dir) if os.path.exists(onnx_node_tests_dir) else []


class NodeTest(OnnxModelInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = onnx_node_tests_dir + self.name + "/model.onnx"

    def construct_inputs(self):
        inputpb = onnx_node_tests_dir + self.name + "/test_data_set_0/input_0.pb"
        return get_tensor_from_pb(inputpb)

    def forward(self, input):
        outputpb = onnx_node_tests_dir + self.name + "/test_data_set_0/output_0.pb"
        return get_tensor_from_pb(outputpb)


for n in names:
    register_test(NodeTest, n)
