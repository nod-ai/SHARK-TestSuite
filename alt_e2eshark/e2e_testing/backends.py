# This file will contain customizations for how to compile mlir from various entrypoints
import abc
from typing import TypeVar
from e2e_testing.storage import TestTensors

CompiledArtifact = TypeVar("CompiledArtifact")
Invoker = TypeVar("Invoker")


class BackendBase(abc.ABC):

    @abc.abstractmethod
    def compile(self, module) -> CompiledArtifact:
        """specifies how to compile an MLIR Module"""

    @abc.abstractmethod
    def load(self, artifact: CompiledArtifact) -> Invoker:
        """loads the compiled artifact"""


from iree import compiler as ic
from iree import runtime as rt
from torch_mlir.passmanager import PassManager


class SimpleIREEBackend(BackendBase):

    def compile(self, module):
        pipeline = "builtin.module(func.func(convert-torch-onnx-to-torch))"
        with module.context as ctx:
            pm = PassManager.parse(pipeline)
            pm.run(module.operation)
        return ic.tools.compile_str(
            str(module), input_type="torch", target_backends=["llvm-cpu"]
        )

    def load(self, artifact):
        config = rt.Config("local-task")
        ctx = rt.SystemContext(config=config)
        vm_module = rt.VmModule.copy_buffer(ctx.instance, artifact)
        ctx.add_vm_module(vm_module)

        def func(x, *, name="main"):
            x = x.data
            device_array = ctx.modules.module[name](*x)
            if isinstance(device_array, tuple):
                np_array = []
                for d in device_array:
                    np_array.append(d.to_host())
                return TestTensors(np_array)
            return TestTensors((device_array.to_host(),))

        return func
