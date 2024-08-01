# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import abc
import onnxruntime as ort
from typing import TypeVar, Union
from e2e_testing.storage import TestTensors

CompiledArtifact = TypeVar("CompiledArtifact")
Invoker = TypeVar("Invoker")

CompiledOutput = Union[CompiledArtifact, ort.InferenceSession]


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module) -> CompiledOutput:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledOutput, func_name: str) -> Invoker:
        """loads the function with name func_name from compiled artifact. This method should return a function callable from python."""


from iree import compiler as ireec
from iree import runtime as ireert


class SimpleIREEBackend(BackendBase):
    '''This backend uses iree to compile and run MLIR modules for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu"):
        self.device = device
        self.hal_target_backend = hal_target_backend

    def compile(self, module, *, save_to: str = None):
        # compile to a vmfb for llvm-cpu
        b = ireec.tools.compile_str(
            str(module),
            target_backends=[self.hal_target_backend],
            extra_args=["--iree-input-demote-i64-to-i32"],
        )
        # log the vmfb
        if save_to:
            with open(save_to + "compiled_model.vmfb", "wb") as f:
                f.write(b)
        return b

    def load(self, artifact, *, func_name="main"):
        config = ireert.Config(self.device)
        ctx = ireert.SystemContext(config=config)
        vm_module = ireert.VmModule.copy_buffer(ctx.instance, artifact)
        ctx.add_vm_module(vm_module)

        def func(x):
            x = x.data
            device_array = ctx.modules.module[func_name](*x)
            if isinstance(device_array, tuple):
                np_array = []
                for d in device_array:
                    np_array.append(d.to_host())
                return TestTensors(np_array)
            return TestTensors((device_array.to_host(),))

        return func


class OnnxrtIreeEpBackend(BackendBase):
    '''This backend uses onnxrt iree-ep to compile and run onnx models for a specified hal_target_backend'''
    def __init__(self, *, device="local-task", hal_target_backend="llvm-cpu"):
        self.device = device
        self.hal_target_backend = hal_target_backend
        self.providers=["IreeExecutionProvider"]
        self.sess_opt = ort.SessionOptions()
        self.sess_opt.execution_mode  = ort.ExecutionMode.ORT_PARALLEL
        #  sess_opt.inter_op_num_threads = 14
        #  sess_opt.intra_op_num_threads = 4
        #  sess_opt.log_verbosity_level = 0

    def compile(self, module, *, save_to: str = None):
        session = ort.InferenceSession(
                   module.model,
                   self.sess_opt,
                   providers=self.providers,
               )
        return session

    def load(self, session, *, func_name=None):
        def func(x):
            output = session.run(None, {input_name: x})
            return TestTensors((output.to_host(),))

        return func
