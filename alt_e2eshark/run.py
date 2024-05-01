import sys
from pathlib import Path

sys.path.append(Path(__file__).parent)
from e2e_testing.storage import TestTensors
from e2e_testing.framework import *
from e2e_testing.registry import GLOBAL_TEST_LIST

test_dir = Path(__file__).parent

# importing the test generating files will register them to
# GLOBAL_TEST_LIST
from onnx_tests.operators import model

# import frontend test configs:
from e2e_testing.test_configs.onnxconfig import OnnxTestConfig

# import backend
from e2e_testing.backends import SimpleIREEBackend


def main():
    # TODO: add argparse to customize config/backend/testlist/etc.
    config = OnnxTestConfig(str(test_dir), SimpleIREEBackend())
    test_list = GLOBAL_TEST_LIST
    run_tests(test_list, config, test_dir)


# TODO: finish and move elsewhere


def run_tests(test_list, config, test_dir):
    # TODO: multi-process
    results = []
    for t in test_list:
        log_dir = str(test_dir) + "/" + t.unique_name
        # TODO: Add logging/log_dir
        inst = t.model_constructor(str(test_dir))
        inputs = inst.construct_inputs()
        golden_outputs = inst.forward(inputs)

        mlir_module = config.mlir_import(inst)
        buffer = config.compile(mlir_module)
        callable_compiled_module = config.backend.load(buffer)
        outputs = callable_compiled_module(inputs)

        result = TestResult(
            name=t.unique_name, input=inputs, gold_output=golden_outputs, output=outputs
        )

        log_result(result, log_dir)


def log_result(result, log_dir):
    summary = summarize_result(result, [1e-4, 1e-4])
    with open(log_dir + ".log", "w+") as f:
        f.write(str(summary))


if __name__ == "__main__":
    main()
