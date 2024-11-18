# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from e2e_testing.registry import register_test
from ..helper_classes import (
    OnnxModelZooDownloadableModel,
    onnx_zoo_validated,
    onnx_zoo_non_validated,
)
from .azure_models import custom_registry

for t in set(onnx_zoo_non_validated).difference(custom_registry):
    t_split = t.split("/")[-2]
    register_test(OnnxModelZooDownloadableModel, t_split)

for t in set(onnx_zoo_validated).difference(custom_registry):
    t_split = ".".join((t.split("/")[-1]).split(".")[:-2])
    register_test(OnnxModelZooDownloadableModel, t_split)
