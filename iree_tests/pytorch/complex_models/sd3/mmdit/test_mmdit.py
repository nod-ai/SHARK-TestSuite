# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest
from ireers import *

# iree_tests/pytorch/complex_models/sdxl/scheduled-unet
current_dir = os.path.dirname(os.path.realpath(__file__))
iree_test_path_extension = os.getenv("IREE_TEST_PATH_EXTENSION", default=str(self.test_cwd))
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
    "--input=2x16x128x128xf16=@inference_input.0.bin",
    "--input=2x154x4096xf16=@inference_input.1.bin",
    "--input=2x2048xf16=@inference_input.2.bin",
    "--input=1xf16=@inference_input.3.bin",
    "--expected_output=2x16x128x128xf32=@inference_output.0.bin"
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

compile_cpu_cmd = get_compile_cmd(mlir_path, "model_cpu.vmfb", flags=CPU_COMPILE_FLAGS)
compile_rocm_cmd = get_compile_cmd(mlir_path, "model_rocm.vmfb", flags=ROCM_COMPILE_FLAGS)

###############################################################################
# CPU
###############################################################################

def test_compile_unet_cpu():
    mlir_path = current_dir + "/model.mlirbc"
    iree_compile(
        mlir_path,
        "model_cpu.vmfb",
        flags=CPU_COMPILE_FLAGS,
        current_dir
    )

def test_run_unet_cpu():
    vmfb_path = current_dir + "/model_cpu.vmfb"
    return iree_run_module(
        vmfb_path,
        flags = [
            "--device=local-task",
            "--parameters=model=real_weights.irpa"
        ] + COMMON_RUN_FLAGS,
        current_dir,
        compile_cpu_cmd
    )

###############################################################################
# ROCM
###############################################################################

def test_compile_unet_rocm():
    mlir_path = current_dir + "/model.mlirbc"
    iree_compile(
        mlir_path,
        "model_rocm.vmfb",
        flags=ROCM_COMPILE_FLAGS,
        current_dir
    )

def test_run_unet_rocm():
    vmfb_path = current_dir + "/model_rocm.vmfb"
    return iree_run_module(
        vmfb_path,
        flags = [
            "--device=hip",
            "--parameters=model=real_weights.irpa"
        ] + COMMON_RUN_FLAGS,
        current_dir,
        compile_rocm_cmd
    )
