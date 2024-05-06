# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path
import argparse
import re

# append alt_e2eshark dir to path to allow importing without explicit pythonpath management
TEST_DIR = Path(__file__).parent
sys.path.append(TEST_DIR)

from e2e_testing.framework import *

# import frontend test configs:
from e2e_testing.test_configs.onnxconfig import OnnxTestConfig

# import backend
from e2e_testing.backends import SimpleIREEBackend


def get_tests(groups):
    """imports tests based on groups specification"""
    combinations = True if groups == "all" or groups == "combinations" else False
    models = True if groups == "all" or groups == "models" else False
    operators = True if groups == "all" or groups == "operators" else False

    from e2e_testing.registry import GLOBAL_TEST_LIST

    # importing the test generating files will register them to GLOBAL_TEST_LIST
    if combinations:
        from onnx_tests.combinations import model
    if models:
        from onnx_tests.models import model
    if operators:
        from onnx_tests.operators import model

    return GLOBAL_TEST_LIST


def main(args):
    """Sets up config and test list based on CL args, then runs the tests"""
    # setup config
    if args.framework != "onnx":
        raise NotImplementedError("only onnx frontend supported now")
    config = OnnxTestConfig(
        str(TEST_DIR), SimpleIREEBackend(hal_target_backend=args.backend)
    )
    # get test list
    # TODO: allow for no-run/mark xfails

    test_list = get_tests(args.groups)
    if args.test_filter:
        test_list = [
            test for test in test_list if re.match(args.test_filter, test.unique_name)
        ]

    # logging setup

    # staging setup

    run_tests(test_list, config, args.rundirectory, args.no_artifacts, args.verbose)


def run_tests(test_list, config, dir_name, no_artifacts, verbose):
    """runs tests in test_list based on config"""
    # TODO: multi-process, argparse, setup exception handling and logging

    parent_log_dir = str(TEST_DIR) + "/" + dir_name + "/"
    # set up a parent log directory to store results
    if not os.path.exists(parent_log_dir):
        os.mkdir(parent_log_dir)
    print(parent_log_dir)

    for t in test_list:
        # set log directory for the individual test
        log_dir = parent_log_dir + t.unique_name + "/"
        print(log_dir)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        # build an instance of the test class
        inst = t.model_constructor(log_dir)

        # generate inputs from the test instance
        inputs = inst.construct_inputs()
        inputs.save_to(log_dir + "input")

        # run native inference
        golden_outputs = inst.forward(inputs)
        golden_outputs.save_to(log_dir + "golden_output")

        artifact_save_to = None if no_artifacts else log_dir
        # generate mlir from the instance using the config
        mlir_module, func_name = config.mlir_import(inst, save_to=artifact_save_to)
        # compile mlir_module using config (calls backend compile)
        buffer = config.compile(mlir_module, save_to=artifact_save_to)
        # run inference with the compiled module
        outputs = config.run(buffer, inputs, func_name=func_name)

        # store outputs
        outputs.save_to(log_dir + "output")

        # model-specific post-processing:
        golden_outputs = inst.apply_postprocessing(golden_outputs)
        outputs = inst.apply_postprocessing(outputs)

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


def _get_argparse():
    msg = "The run.py script to run e2e shark tests"
    parser = argparse.ArgumentParser(prog="run.py", description=msg, epilog="")

    # test config related arguments:
    parser.add_argument(
        "-b",
        "--backend",
        choices=["llvm-cpu", "amd-aie", "rocm", "vulkan"],
        default="llvm-cpu",
        help="Target backend i.e. hardware to run on",
    )
    parser.add_argument(
        "-f",
        "--framework",
        nargs="*",
        choices=["pytorch", "onnx", "tensorflow"],
        default="onnx",
        help="Run tests for given framework(s)",
    )
    parser.add_argument(
        "--torchtolinalg",
        action="store_true",
        default=False,
        help="Have torch-mlir-opt to produce linalg instead of torch mlir and pass that to iree-compile",
    )
    parser.add_argument(
        "--torchmlirimport",
        choices=["compile", "fximport"],
        default="fximport",
        help="Use torch_mlir.torchscript.compile, or Fx importer",
    )

    # arguments for customizing test-stages:
    parser.add_argument(
        "--runfrom",
        choices=["model-run", "torch-mlir", "iree-compile"],
        default="model-run",
        help="Start from model-run, or torch MLIR, or IREE compiled artefact",
    )
    parser.add_argument(
        "--runupto",
        choices=["torch-mlir", "iree-compile", "inference"],
        default="inference",
        help="Run upto torch MLIR generation, IREE compilation, or inference",
    )

    # test-list filtering arguments:
    parser.add_argument(
        "-g",
        "--groups",
        nargs="*",
        choices=["operators", "combinations", "models"],
        default="all",
        help="Run given group of tests",
    )
    parser.add_argument(
        "-t",
        "--test-filter",
        nargs="*",
        help="Run given specific test(s) only",
    )
    parser.add_argument(
        "--testsfile",
        help="A file with lists of tests (starting with framework name) to run",
    )

    # test tolerance
    parser.add_argument(
        "--tolerance",
        help="Set abolsulte (atol) and relative (rtol) tolerances for comparing floating point numbers. Example: --tolerance 1e-03 1-04",
        nargs="+",
        type=float,
    )

    # logging
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print aditional messsages to show progress",
    )
    parser.add_argument(
        "-r",
        "--rundirectory",
        default="test-run",
        help="The test run directory",
    )
    parser.add_argument(
        "--no-artifacts",
        help="If enabled, this flag will prevent saving mlir and vmfb files",
        action="store_true",
        default=False,
    )
    # parser.add_argument(
    #     "--cachedir",
    #     help="Please select a dir with large free space to cache all torch, hf, turbine_tank model data",
    #     required=True,
    # )
    # parser.add_argument(
    #     "-d",
    #     "--todtype",
    #     choices=["default", "fp32", "fp16", "bf16"],
    #     default="default",
    #     help="If not default, casts model and input to given data type if framework supports model.to(dtype) and tensor.to(dtype)",
    # )
    # parser.add_argument(
    #     "-j",
    #     "--jobs",
    #     type=int,
    #     default=4,
    #     help="Number of parallel processes to use per machine for running tests",
    # )
    # parser.add_argument(
    #     "-m",
    #     "--mode",
    #     choices=["direct", "turbine", "onnx", "ort", "vaiml"],
    #     default="onnx",
    #     help="direct=Fx/TS->torch-mlir, turbine=aot-export->torch-mlir, onnx=exportonnx-to-torch-mlir, ort=exportonnx-to-ortep",
    # )
    # parser.add_argument(
    #     "--norun",
    #     action="store_true",
    #     default=False,
    #     help="Skip running of tests. Useful for generating test summary after the run",
    # )
    # parser.add_argument(
    #     "--report",
    #     action="store_true",
    #     default=False,
    #     help="Generate test report summary",
    # )
    # parser.add_argument(
    #     "--reportformat",
    #     choices=["pipe", "github", "html", "csv"],
    #     default="pipe",
    #     help="Format of the test report summary file. It takes subset of tablefmt value of python tabulate",
    # )
    # parser.add_argument(
    #     "--uploadtestsfile",
    #     help="A file with lists of tests that should be uploaded",
    # )
    return parser


if __name__ == "__main__":
    parser = _get_argparse()
    main(parser.parse_args())
