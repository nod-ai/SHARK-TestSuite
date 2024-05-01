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


class OnnxTestConfig(TestConfig):

    def __init__(self, log_dir: str, backend: BackendBase):
        super().__init__()
        self.log_dir = log_dir
        self.backend = backend

    def mlir_import(self, model_info: OnnxModelInfo):
        model = onnx.load(model_info.model)
        model = onnx.shape_inference.infer_shapes(model)

        context = Context()
        torch_d.register_dialect(context)
        model_info = onnx_importer.ModelInfo(model)
        m = model_info.create_module(context=context)
        imp = onnx_importer.NodeImporter.define_function(
            model_info.main_graph, m.operation
        )
        imp.import_all()
        return m

    def compile(self, mlir_module, *, path: str = None):
        return self.backend.compile(mlir_module, path=path)

    def run(self, artifact, inputs):
        func = self.backend.load(artifact)
        return func(inputs)
