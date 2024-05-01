import onnxruntime
import numpy, torch
import abc
import os
from typing import Union, TypeVar, Tuple, NamedTuple, Dict, Optional, Callable
from e2e_testing.storage import TestTensors

Module = TypeVar("Module")

def generate_input_from_node(node: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg):
    if node.type == "tensor(float)":
        return numpy.random.randn(*node.shape).astype(numpy.float32)
    if node.type == "tensor(int)":
        return numpy.random.randint(0, 10000, size=node.shape).astype(numpy.int32)
    if node.type == "tensor(bool)":
        return numpy.random.randint(0, 2, size=node.shape).astype(bool) 

def get_sample_inputs_for_onnx_model(model_path):
    s = onnxruntime.InferenceSession(model_path, None)
    inputs = s.get_inputs()
    sample_inputs = TestTensors(tuple([generate_input_from_node(node) for node in inputs]))
    return sample_inputs 

class OnnxModelInfo:
    '''Stores information about an onnx test: the filepath to model.onnx, and how to construct/download it.'''
    def __init__(self, onnx_model_path: str, dim_params: Optional[Dict[str, int]] = None):
        self.model = onnx_model_path

    def forward(self, input: Optional[TestTensors] = None) -> TestTensors:
        '''Applies self.model to self.input'''
        input = input.to_numpy().data
        if not os.path.exists(self.model):
            self.construct_model()
        session = onnxruntime.InferenceSession(self.model, None)
        session_inputs = session.get_inputs()
        session_outputs = session.get_outputs()

        model_output = session.run(
            [output.name for output in session_outputs],
            {session_inputs[i].name: input[i] for i in range(len(session_inputs))},
        )

        return TestTensors(model_output)
    
    def construct_model(self):
        '''a method to be overwritten'''
        raise Exception(f'Model path {self.model} does not exist and no construct_model method is defined.')

    def construct_inputs(self):
        '''can be over-written to generate specific inputs'''
        if not os.path.exists(self.model):
            self.construct_model()
        return get_sample_inputs_for_onnx_model(self.model)

TestModel = Union[OnnxModelInfo, torch.nn.Module]

CompiledArtifact = TypeVar("CompiledArtifact")

class TestConfig(abc.ABC):

    @abc.abstractmethod
    def mlir_import(self, program: TestModel) -> Module:
        '''imports the test model to an MLIR Module'''

    @abc.abstractmethod
    def compile(self, mlir_module: Module) -> CompiledArtifact:
        '''converts the test program to a compiled artifact'''
        pass

    @abc.abstractmethod
    def run(self, artifact: CompiledArtifact, input: TestTensors) -> TestTensors:
        '''runs the input through the compiled artifact'''
        pass

class Test(NamedTuple):
    unique_name: str
    model_constructor: Callable[[], TestModel]


class TestResult(NamedTuple):
    name: str
    input: TestTensors
    gold_output: TestTensors
    output: TestTensors

def summarize_result(test_result: TestResult, tol):
    output = test_result.output.to_torch().data
    gold = test_result.gold_output.to_torch().data
    if len(output) != len(gold):
        raise Exception(f"num outputs: {len(output)} doesn't match num golden: {len(gold)} for test {test_result.name}")
    match = []
    for i in range(len(output)):
        match.append(torch.isclose(output[i].to(dtype= gold[i].dtype),gold[i], *tol))
    return TestTensors(tuple(match))
