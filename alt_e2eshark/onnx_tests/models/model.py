# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import os
import numpy
import onnx
import torch
from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed, fix_output_shapes
from e2e_testing import azutils
from e2e_testing.framework import OnnxModelInfo
from e2e_testing.registry import register_test
from e2e_testing.storage import TestTensors
from .protected_list import protected_models, models_with_postprocessing

class AzureDownloadableModel(OnnxModelInfo):
    def __init__(self, name: str, onnx_model_path: str, cache_dir: str):
        opset_version = 21
        super().__init__(name, onnx_model_path, cache_dir, opset_version)

    def construct_model(self):
        # try to find a .onnx file in the test-run dir
        # if that fails, check for zip file in cache
        # if that fails, try to download and setup from azure, then search again for a .onnx file

        # TODO: make the zip file structure more uniform so we don't need to search for extracted files
        model_dir = self.model.rstrip("model.nx")

        def find_models(model_dir):
            # search for a .onnx file in the ./test-run/testname/ dir
            found_models = []
            for root, dirs, files in os.walk(model_dir):
                for name in files:
                    if name[-5:] == ".onnx":  
                        found_models.append(os.path.abspath(os.path.join(root, name)))
            return found_models

        found_models = find_models(model_dir) 

        if len(found_models) == 0:
            azutils.pre_test_onnx_model_azure_download(
                self.name, self.cache_dir, self.model
            )
            found_models = find_models(model_dir)
        if len(found_models) == 1:
            self.model = found_models[0]
            return
        if len(found_models) > 1:
            print(f'Found multiple model.onnx files: {found_models}')
            print(f'Picking the first model found to use: {found_models[0]}')
            self.model = found_models[0]
            return
        raise OSError(f"No onnx model could be found, downloaded, or extracted to {model_dir}")

    def apply_postprocessing(self, output: TestTensors):
        if self.name not in models_with_postprocessing:
            return output
        processed_outputs = []
        for d in output.to_torch().data:
            processed_outputs.append(
                torch.sort(torch.topk(torch.nn.functional.softmax(d, 1), 2)[1])[0]
            )
        return TestTensors(processed_outputs)

for t in protected_models:
    register_test(AzureDownloadableModel, t)


register_test(AzureDownloadableModel, "RAFT_vaiq_int8")
register_test(AzureDownloadableModel, "pytorch-3dunet_vaiq_int8")
register_test(AzureDownloadableModel, "FCN_vaiq_int8")
register_test(AzureDownloadableModel, "u-net_brain_mri_vaiq_int8")


class SampleModelWithPostprocessing(AzureDownloadableModel):
    def apply_postprocessing(self, output: TestTensors):
        processed_outputs = []
        for d in output.to_torch().data:
            processed_outputs.append(
                torch.sort(torch.topk(torch.nn.functional.softmax(d, 1), 2)[1])[0]
            )
        return TestTensors(processed_outputs)


register_test(SampleModelWithPostprocessing, "resnet50_vaiq_int8")


class Opt125MAWQModelInfo(AzureDownloadableModel):
    def set_model_params(self, batch_size, sequence_length, past_sequence_length):
        self.dim_params = (
            "batch_size",
            "sequence_length",
            "past_sequence_length",
            "past_sequence_length + 1",
        )
        self.dim_values = (
            batch_size,
            sequence_length,
            past_sequence_length,
            past_sequence_length + 1,
        )

    def construct_inputs(self):
        self.set_model_params(1, 1, 0)
        pv_zip = zip(self.dim_params, self.dim_values)
        pv = dict(pv_zip)

        model_inputs = [
            numpy.random.randint(
                -1000,
                high=1000,
                size=(pv["batch_size"], pv["sequence_length"]),
                dtype=numpy.int64,
            )
        ]  # input_ids
        model_inputs.append(
            numpy.random.randint(
                -10,
                high=10,
                size=(pv["batch_size"], pv["past_sequence_length + 1"]),
                dtype=numpy.int64,
            )
        )  # attention_mask
        for i in range(2 * 12):
            model_inputs.append(
                numpy.random.rand(
                    pv["batch_size"], 12, pv["past_sequence_length"], 64
                ).astype(numpy.float32)
            )  # 12 key/value pairs
        return TestTensors(model_inputs)


register_test(Opt125MAWQModelInfo, "opt-125M-awq")


# hugging face example:
class Opt125MGPTQModelInfo(OnnxModelInfo):
    def construct_model(self):
        # model origin: https://huggingface.co/jlsilva/facebook-opt-125m-gptq4bit
        test_modelname = "facebook/opt-125m"
        quantizedmodelname = "jlsilva/facebook-opt-125m-gptq4bit"
        kwargs = {
            "torch_dtype": torch.float32,
            "trust_remote_code": True,
        }
        from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

        quantization_config = GPTQConfig(bits=8, disable_exllama=True)
        kwargs["quantization_config"] = quantization_config
        kwargs["device_map"] = "cpu"
        self.pytorch_model = AutoModelForCausalLM.from_pretrained(
            quantizedmodelname, **kwargs
        )
        # model.output_hidden_states = False
        self.tokenizer = AutoTokenizer.from_pretrained(test_modelname)
        self.prompt = "What is nature of our existence?"
        self.encoding = self.tokenizer(self.prompt, return_tensors="pt")
        torch.onnx.export(
            self.pytorch_model,
            (self.encoding["input_ids"], self.encoding["attention_mask"]),
            self.model,
            opset_version=20,
        )

    def construct_inputs(self):
        if not self.__dict__.__contains__("tokenizer"):
            self.construct_model()
        return TestTensors(
            (self.encoding["input_ids"], self.encoding["attention_mask"])
        )

    def forward(self, input):
        response = self.pytorch_model.generate(
            input.data[0],
            do_sample=True,
            top_k=50,
            max_length=100,
            top_p=0.95,
            temperature=1.0,
        )
        return TestTensors((response,))

        # model_response = model.generate(
        #     E2ESHARK_CHECK["input"],
        #     do_sample=True,
        #     top_k=50,
        #     max_length=100,
        #     top_p=0.95,
        #     temperature=1.0,
        # )
        # print("Response:", tokenizer.decode(model_response[0]))


register_test(Opt125MGPTQModelInfo, "opt-125m-gptq")


# this is to sample adding a static version of a test
# class Opt125MAWQStaticModel(Opt125MAWQModelInfo):
#     def construct_model(self):
#         self.set_model_params(1,1,0)
#         pv_zip = zip(self.dim_params, self.dim_values)
#         pv = dict(pv_zip)
#         model = onnx.load(self.model.rstrip("/model.onnx").rstrip(self.name) + "opt-125M-awq/onnx/models/opt-125M-awq/model.onnx")
#         for p in self.dim_params:
#             make_dim_param_fixed(model.graph, p, pv[p])
#         fix_output_shapes(model)
#         onnx.save(model, self.model)
# register_test(Opt125MAWQStaticModel, "static-opt-125M-awq")
