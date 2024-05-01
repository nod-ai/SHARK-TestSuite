# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


def run_tests(test_list, config, test_dir):
    # TODO: multi-process, argparse, setup exception handling and logging

    # set up a parent log directory to store results
    parent_log_dir = str(test_dir) + "/test-run/"
    if not os.path.exists(parent_log_dir):
        os.mkdir(parent_log_dir)

    for t in test_list:
        # set log directory for this test
        log_dir = parent_log_dir + t.unique_name + "/"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # build an instance of the test class
        inst = t.model_constructor(log_dir)

        # generate inputs from the test instance
        inputs = inst.construct_inputs()

        # run native inference
        golden_outputs = inst.forward(inputs)

        # generate mlir from the instance using the config
        mlir_module = config.mlir_import(inst)
        with open(log_dir + "import.mlir", "w") as f:
            f.write(str(mlir_module))
        # compile mlir_module using config (calls backend compile)
        buffer = config.compile(mlir_module)
        # explicitly call backend load method to get the forward function as a python callable
        callable_compiled_module = config.backend.load(buffer)
        # run the inputs through the loaded callable
        outputs = callable_compiled_module(inputs)

        # store the results
        result = TestResult(
            name=t.unique_name, input=inputs, gold_output=golden_outputs, output=outputs
        )
        # log the results
        log_result(result, log_dir, [1e-4, 1e-4])


def log_result(result, log_dir, tol):
    summary = summarize_result(result, tol)
    with open(log_dir + "inference_comparison.log", "w+") as f:
        f.write(f"matching values with (rtol,atol) = {tol}: {summary}\n{result}")


if __name__ == "__main__":
    main()
