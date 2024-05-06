# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import onnx
from torch_mlir.extras import onnx_importer
from torch_mlir.dialects import torch as torch_d
from torch_mlir.ir import Context
from e2e_testing.backends import BackendBase
from e2e_testing.framework import TestConfig, OnnxModelInfo
from torch_mlir.passmanager import PassManager


class OnnxTestConfig(TestConfig):

    def __init__(self, log_dir: str, backend: BackendBase):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend

    def mlir_import(self, model_info: OnnxModelInfo, *, save_to: str = None):
        model = onnx.load(model_info.model)
        shaped_model = onnx.shape_inference.infer_shapes(model, data_prop=True)
        func_name = shaped_model.graph.name
        context = Context()
        torch_d.register_dialect(context)
        model_info = onnx_importer.ModelInfo(shaped_model)
        m = model_info.create_module(context=context)
        imp = onnx_importer.NodeImporter.define_function(
            model_info.main_graph, m.operation
        )
        imp.import_all()
        # log imported IR
        if save_to:
            with open(save_to + "model.torch_onnx.mlir", "w") as f:
                f.write(str(m))
        # convert imported torch-onnx ir to torch
        pipeline = "builtin.module(func.func(convert-torch-onnx-to-torch))"
        with m.context as ctx:
            pm = PassManager.parse(pipeline)
            pm.run(m.operation)
        # log torch-mlir IR
        if save_to:
            with open(save_to + "model.torch.mlir", "w") as f:
                f.write(str(m))
        return m, func_name

    def compile(self, mlir_module, *, save_to: str = None):
        return self.backend.compile(mlir_module, save_to=save_to)

    def run(self, artifact, inputs, *, func_name="main"):
        func = self.backend.load(artifact, func_name=func_name)
        return func(inputs)
