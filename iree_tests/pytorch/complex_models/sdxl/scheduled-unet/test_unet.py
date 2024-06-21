# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from iree_tests.pytorch.complex_models.ireers import *
import os
import setuptools

print(setuptools.find_packages())

# iree_tests/pytorch/complex_models/sdxl/scheduled-unet
current_dir = os.path.dirname(os.path.realpath(__file__))
iree_test_path_extension = os.getenv("IREE_TEST_PATH_EXTENSION", default=current_dir)
rocm_chip = os.getenv("ROCM_CHIP", default="gfx90a")

###############################################################################
# Fixtures
###############################################################################

CPU_COMPILE_FLAGS = [
    "--iree-hal-target-backends=llvm-cpu",
    "--iree-llvmcpu-target-cpu-features=host",
    "--iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false",
    "--iree-llvmcpu-distribution-size=32",
    "--iree-opt-const-eval=false",
    "--iree-llvmcpu-enable-ukernels=all",
    "--iree-global-opt-enable-quantized-matmul-reassociation"
]

COMMON_RUN_FLAGS = [
    "--input=1x4x128x128xf16=@inference_input.0.bin",
    "--input=2x64x2048xf16=@inference_input.1.bin",
    "--input=2x1280xf16=@inference_input.2.bin",
    "--input=1xf16=@inference_input.3.bin",
    "--expected_output=1x4x128x128xf16=@inference_output.0.bin"
]

ROCM_COMPILE_FLAGS = [
    "--iree-hal-target-backends=rocm",
    f"--iree-rocm-target-chip={rocm_chip}",
    "--iree-opt-const-eval=false",
    f"--iree-codegen-transform-dialect-library={iree_test_path_extension}/attention_and_matmul_spec.mlir",
    "--iree-global-opt-propagate-transposes=true",
    "--iree-global-opt-enable-fuse-horizontal-contractions=true",
    "--iree-flow-enable-aggressive-fusion=true",
    "--iree-opt-aggressively-propagate-transposes=true",
    "--iree-opt-outer-dim-concat=true",
    "--iree-vm-target-truncate-unsupported-floats",
    "--iree-llvmgpu-enable-prefetch=true",
    "--iree-opt-data-tiling=false",
    "--iree-codegen-gpu-native-math-precision=true",
    "--iree-codegen-llvmgpu-use-vector-distribution",
    "--iree-rocm-waves-per-eu=2",
    "--iree-execution-model=async-external",
    "--iree-preprocessing-pass-pipeline=builtin.module(iree-preprocessing-transpose-convolution-pipeline, util.func(iree-preprocessing-pad-to-intrinsics))"
]

mlir_path = current_dir + "/model.mlirbc"
compile_cpu_cmd = get_compile_cmd(mlir_path, "model_cpu.vmfb", CPU_COMPILE_FLAGS)
compile_rocm_cmd = get_compile_cmd(mlir_path, "model_rocm.vmfb", ROCM_COMPILE_FLAGS)

###############################################################################
# CPU
###############################################################################

def test_compile_unet_cpu():
    iree_compile(
        mlir_path,
        "model_cpu.vmfb",
        CPU_COMPILE_FLAGS,
        current_dir
    )

def test_run_unet_cpu():
    vmfb_path = current_dir + "/model_cpu.vmfb"
    return iree_run_module(
        vmfb_path,
        [
            "--device=local-task",
            "--parameters=model=real_weights.irpa",
            "--module=sdxl_scheduled_unet_pipeline_fp16_cpu.vmfb",
            "--expected_f16_threshold=0.8f"
        ] + COMMON_RUN_FLAGS,
        current_dir,
        compile_cpu_cmd
    )

###############################################################################
# ROCM
###############################################################################

def test_compile_unet_rocm():
    iree_compile(
        mlir_path,
        "model_rocm.vmfb",
        ROCM_COMPILE_FLAGS,
        current_dir
    )

def test_run_unet_rocm():
    vmfb_path = current_dir + "/model_rocm.vmfb"
    return iree_run_module(
        vmfb_path,
        [
            "--device=hip",
            "--parameters=model=real_weights.irpa",
            "--module=sdxl_scheduled_unet_pipeline_fp16_rocm.vmfb",
            "--expected_f16_threshold=0.7f"
        ] + COMMON_RUN_FLAGS,
        current_dir,
        compile_rocm_cmd
    )
