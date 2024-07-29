# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import torch
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from .azure_models import AzureDownloadableModel
import os

class ImageClassificationModel(OnnxModelInfo):
    def __init__(self, name_appendix, *args, **kwargs):
        self.name_appendix = name_appendix
        super().__init__(*args, **kwargs)

    def construct_model(self):
        og_name = self.name.rstrip(self.name_appendix)
        og_model_path = self.model.rstrip(self.name_appendix + "/model.onnx") + "/model.onnx"
        inst = AzureDownloadableModel(og_name,og_model_path,self.cache_dir)
        inst.construct_model()
        self.model = inst.model

    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            processed_outputs.append(
                torch.sort(torch.topk(torch.nn.functional.softmax(d, 1), 2)[1])[0]
            )
        return TestTensors(processed_outputs)

register_test(AzureDownloadableModel, "resnet50_vaiq_int8")
register_test(lambda *args, **kwargs : ImageClassificationModel("pp", *args, **kwargs), "resnet50_vaiq_int8pp")