# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import onnx

# TODO: change this to use command-line args for optype and depth

def load():
    raw_model = onnx.load("model.onnx")
    model = onnx.shape_inference.infer_shapes(raw_model, data_prop=True)
    return model

def modify_model(model, new_output_name: str):
    print(f"original output = {model.graph.output[0].name}")
    for vi in model.graph.value_info:
        # specify the name of the value info you want to have as an output
        if vi.name == new_output_name:
            model.graph.output.pop()
            model.graph.output.append(vi)
    print(f"new output = {model.graph.output[0].name}")
    onnx.save(model,"modified_model.onnx")

def node_output_name(model, n: int, op_name: str):
    '''returns the output name for the nth node in the onnx model with op_type given by op_name'''
    counter=0
    for nde in model.graph.node:
        if nde.op_type != op_name:
            continue
        if counter == n:
            node = nde
            print(nde)
        counter += 1
    if not node:
        print("failed to find node")
        import sys
        sys.exit(1)
    return node.output[0]

model = load()
modify_model(model, node_output_name(model, 2, "Conv"))
import os
