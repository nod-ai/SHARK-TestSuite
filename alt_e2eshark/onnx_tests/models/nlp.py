# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from ..helper_classes import AzureDownloadableModel
from e2e_testing.registry import register_test
from e2e_testing.storage import load_test_txt_file
from e2e_testing.onnx_utils import get_node_shape_from_dim_param_dict
from e2e_testing.storage import TestTensors
import onnxruntime as ort
import numpy
from typing import Optional
import onnx

this_file = Path(__file__)
lists_dir = (this_file.parent).joinpath("external_lists")

model_names = []
for i in [1, 2, 3]:
    model_names += load_test_txt_file(lists_dir.joinpath(f"nlp-shard{i}.txt"))

large_file_size_models = [
    "model--YuisekinAI-mistral-0.7B--yuiseki",
    "model--financial-summarization-pegasus--human-centered-summarization",
    "model--finetuned_gpt2-large_sst2_negation0.0001_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs2--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.001_pretrainedTrue_epochs3--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.01--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.01_pretrainedFalse_epochs10--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.01_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.05--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.0_pretrainedFalse--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.1_pretrainedFalse_epochs10--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.1_pretrainedTrue_epochs1--jhaochenz",
    "model--finetuned_gpt2-large_sst2_negation0.2--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.5--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.5_pretrainedFalse--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.8--yuhuizhang",
    "model--finetuned_gpt2-large_sst2_negation0.8_pretrainedFalse--yuhuizhang",
    "model--flan-t5-large-samsum--oguuzhansahin",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP13--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP14--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP15--pszemraj",
    "model--long-t5-tglobal-large-pubmed-3k-booksum-16384-WIP17--pszemraj",
    "model--m2m100_418M-finetuned-kde4-en-to-pt_BR--danhsf",
    "model--m2m100_418M-fr--Jour",
    "model--m2m100_418M-fr--NDugar",
    "model--m2m100_418M-ja--vivek-307306",
    "model--mT5-base-HunSum-1--SZTAKI-HLT",
    "model--mT5_multilingual_XLSum--csebuetnlp",
    "model--manifestoberta-xlm-roberta-56policy-topics-sentence-2023-1-1--manifesto-project",
    "model--my_xlm-roberta-large-finetuned-conll03--BahAdoR0101",
    "model--pegasus-cnn_dailymail--google",
    "model--pegasus-large-book-summary--pszemraj",
    "model--pegasus-large-booksum--cnicu",
    "model--pegasus-large-summary-explain--pszemraj",
    "model--pegasus-xsum--google",
    "model--pegasus_summarizer--tuner007",
    "model--roberta-ner-multilingual--julian-schelb",
    "model--t5-large-finetuned-xsum-cnn--sysresearch101",
    "model--tglobal-large-booksum-WIP4-r1--pszemraj",
    "model--xlm-roberta-large-squad2--deepset",
    "model--xlmr-large-qa-fa--m3hrdadfi",
]


def dim_param_constructor(dim_param_dict):
    class AzureWithDimParams(AzureDownloadableModel):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.name in large_file_size_models:
                self.construct_model()
                self.update_model_without_ext_data()
                self.opset_version = None
            # TODO: check opset versions
            # print(f"Opset: {self.opset_version}")
            # self.update_opset_version_and_overwrite()

        def update_dim_param_dict(self):
            self.dim_param_dict = dim_param_dict

        def construct_inputs(self):
            """Overrides the parent class method to construct sample inputs with the correct dimensions."""
            default_inputs = super().construct_inputs()

            tensors = list(default_inputs.data)

            self.update_sess_options()
            session = ort.InferenceSession(self.model, self.sess_options)

            # nlp specific overrides
            rng = numpy.random.default_rng(19)
            for i, node in enumerate(session.get_inputs()):
                if node.name == "token_type_ids" or node.name == "attention_mask":
                    int_dims = get_node_shape_from_dim_param_dict(
                        node, self.dim_param_dict
                    )
                    tensors[i] = rng.integers(0, 2, size=int_dims, dtype=numpy.int64)
            if self.name == "model--s2t-medium-librispeech-asr--facebook":
                tensors[1] = rng.integers(
                    0, 2, size=tensors[1].shape, dtype=numpy.int64
                )
            default_sample_inputs = TestTensors(tuple(tensors))
            return default_sample_inputs

    return AzureWithDimParams


# Default dimension parameters for NLP models

# TODO: Verify these dim params are valid for each model, or load them from a json
default_nlp_params = {
    "batch_size": 1,
    "seq_len": 128,
    "encoder_sequence_length": 128,
    "decoder_sequence_length": 128,
}
dim_aliases = [
    {"seq_len", "sequence_length"},
]
for alias_set in dim_aliases:
    found = set(alias_set).intersection(default_nlp_params.keys())
    if len(found) > 1:
        # check if the values are the same
        val = default_nlp_params[next(iter(found))]
        if not all(default_nlp_params[alias] == val for alias in found):
            raise ValueError(
                f"Multiple aliases for the same dimension have different values: {found}"
            )

    aliases = alias_set - found
    for alias in aliases:
        default_nlp_params[alias] = default_nlp_params[next(iter(found))]

# Custom dimension parameters for specific models
custom_dim_params = {
    # Add custom dimension parameters for specific models here
    # Example:
    # "model_name": {"batch_size": 1, "seq_len": 256, "custom_dim": 64},
    "model--s2t-medium-librispeech-asr--facebook": {
        "batch_size": 1,
        "feature_size": 80,
        "encoder_sequence_length": 80,
        "decoder_sequence_length": 80,
    },
}

default_param_models = set(model_names).difference(set(custom_dim_params.keys()))

# Register models with default parameters
for model_name in default_param_models:
    register_test(dim_param_constructor(default_nlp_params), model_name)


# Register models with custom parameters
for model_name, dim_params in custom_dim_params.items():
    register_test(dim_param_constructor(dim_params), model_name)

# You can add more customizations or specific handling for certain models here
from ..helper_classes import TruncatedModel, get_trucated_constructor
# download model.onnx to /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/tmp/mygpt4
# export CACHE_DIR=/proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/tmp
model_name = "mygpt4"
mygpt4_nlp_params = {
    "unk__2829": 1,
    "unk__2828": 1,
    "unk__2827": 1,
    "unk__2826": 1,
    "unk__2825": 1,
    "unk__2824": 1,
}
register_test(dim_param_constructor(mygpt4_nlp_params), "mygpt4")
t_model_constructor = get_trucated_constructor(TruncatedModel, dim_param_constructor(mygpt4_nlp_params), model_name)
# Run to the last onnx ops / the whole model with:
# python ./run.py --mode=cl-onnx-iree -v -t mygpt4 --torchtolinalg
#   Stages to be run: ['setup', 'import_model', 'preprocessing', 'compilation', 'construct_inputs', 'native_inference', 'compiled_inference', 'postprocessing']
#   Test list: ['mygpt4']
#   running test mygpt4...
#   Unzipping - /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/tmp/mygpt4/model.onnx.zip... 
#   Unzipping succeded. Look for extracted contents in /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/test-run/mygpt4
#   {'Shape': 63, 'Cast': 141, 'Slice': 136, 'Squeeze': 25, 'Range': 25, 'Concat': 65, 'Reshape': 161, 'Unsqueeze': 38, 'Sub': 85, 'Mul': 169, 'Add': 159, 'Gather': 3, 'MatMul': 72, 'Split': 12, 'Transpose': 48, 'GreaterOrEqual': 12, 'Softmax': 12, 'GlobalAveragePool': 48, 'Sqrt': 24, 'Reciprocal': 24, 'Erf': 12}#         
#   Running stage 'import_model'...      
register_test(t_model_constructor(1, ""), "mygpt4_1")
# Run to the last 2/7/12/17 onnx op with:
# run with python ./run.py --mode=cl-onnx-iree -v -t mygpt4_trunc_
# for n in range(2, 20, 5):
#     register_test(t_model_constructor(n,""), f"mygpt4_trunc_{n}")

# Run to 5/55/105/155 onnx.Add ops with:
# run with python ./run.py --mode=cl-onnx-iree -v -t mygpt4_trunc_add_
# for n in range(5, 160, 50):
#     register_test(t_model_constructor(n,"Add"), f"mygpt4_trunc_add_{n}")

# python ./run.py --mode=cl-onnx-iree -v -t mygpt4_trunc_shape_1
register_test(t_model_constructor(1, "Shape"), "mygpt4_trunc_shape_1")
# register_test(t_model_constructor(160, "Reshape"), "mygpt4_trunc_reshape_160")
for n in range(0, 160, 20):
    register_test(t_model_constructor(n,"Reshape"), f"mygpt4_trunc_Reshape_{n}")
# python ./run.py --mode=cl-onnx-iree -v -t mygpt4 --torchtolinalg
# Test list: ['mygpt4', 'mygpt4_1', 'mygpt4_trunc_Reshape_0', 'mygpt4_trunc_Reshape_100', 'mygpt4_trunc_Reshape_120', 'mygpt4_trunc_Reshape_140', 'mygpt4_trunc_Reshape_20', 'mygpt4_trunc_Reshape_40', 'mygpt4_trunc_Reshape_60', 'mygpt4_trunc_Reshape_80', 'mygpt4_trunc_shape_1']

# python ./run.py --mode=cl-onnx-iree -v --torchtolinalg -t mygpt4                                                       
# Stages to be run: ['setup', 'import_model', 'preprocessing', 'compilation', 'construct_inputs', 'native_inference', 'compiled_inference', 'postprocessing']
# Test list: ['mygpt4', 'mygpt4_1', 'mygpt4_trunc_Reshape_0', 'mygpt4_trunc_Reshape_100', 'mygpt4_trunc_Reshape_120', 'mygpt4_trunc_Reshape_140', 'mygpt4_trunc_Reshape_20', 'mygpt4_trunc_Reshape_40', 'mygpt4_trunc_Reshape_60', 'mygpt4_trunc_Reshape_80', 'mygpt4_trunc_shape_1']
# running test mygpt4...
# Unzipping - /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/tmp/mygpt4/model.onnx.zip... 
# Unzipping succeded. Look for extracted contents in /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/test-run/mygpt4
#         FAILED (compilation)                    
# running test mygpt4_1...
# {'Shape': 63, 'Cast': 141, 'Slice': 136, 'Squeeze': 25, 'Range': 25, 'Concat': 65, 'Reshape': 161, 'Unsqueeze': 38, 'Sub': 85, 'Mul': 169, 'Add': 159, 'Gather': 3, 'MatMul': 72, 'Split': 12, 'Transpose': 48, 'GreaterOrEqual': 12, 'Softmax': 12, 'GlobalAveragePool': 48, 'Sqrt': 24, 'Reciprocal': 24, 'Erf': 12}
#         FAILED (native_inference)                    
# running test mygpt4_trunc_Reshape_0...
# {'Shape': 1, 'Cast': 2, 'Slice': 1, 'Concat': 1, 'Reshape': 1}
#         FAILED (compiled_inference)                    
# running test mygpt4_trunc_Reshape_100...
# {'Shape': 40, 'Cast': 90, 'Slice': 86, 'Squeeze': 17, 'Range': 17, 'Concat': 41, 'Reshape': 100, 'Unsqueeze': 26, 'Sub': 51, 'Mul': 100, 'Add': 96, 'Gather': 3, 'MatMul': 44, 'Split': 8, 'Transpose': 30, 'GreaterOrEqual': 8, 'Softmax': 7, 'GlobalAveragePool': 28, 'Sqrt': 14, 'Reciprocal': 14, 'Erf': 7}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_120...
# {'Shape': 48, 'Cast': 107, 'Slice': 103, 'Squeeze': 19, 'Range': 19, 'Concat': 49, 'Reshape': 121, 'Unsqueeze': 29, 'Sub': 62, 'Mul': 123, 'Add': 117, 'Gather': 3, 'MatMul': 54, 'Split': 9, 'Transpose': 36, 'GreaterOrEqual': 9, 'Softmax': 9, 'GlobalAveragePool': 34, 'Sqrt': 17, 'Reciprocal': 17, 'Erf': 9}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_140...
# {'Shape': 55, 'Cast': 123, 'Slice': 119, 'Squeeze': 23, 'Range': 23, 'Concat': 56, 'Reshape': 141, 'Unsqueeze': 35, 'Sub': 74, 'Mul': 144, 'Add': 136, 'Gather': 3, 'MatMul': 63, 'Split': 11, 'Transpose': 44, 'GreaterOrEqual': 11, 'Softmax': 11, 'GlobalAveragePool': 40, 'Sqrt': 20, 'Reciprocal': 20, 'Erf': 10}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_20...
# {'Shape': 9, 'Cast': 21, 'Slice': 17, 'Squeeze': 3, 'Range': 3, 'Concat': 10, 'Reshape': 20, 'Unsqueeze': 5, 'Sub': 8, 'Mul': 15, 'Add': 17, 'Gather': 3, 'MatMul': 7, 'Split': 2, 'Transpose': 4, 'GreaterOrEqual': 1, 'Softmax': 1, 'GlobalAveragePool': 4, 'Sqrt': 2, 'Reciprocal': 2, 'Erf': 1}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_40...
# {'Shape': 17, 'Cast': 39, 'Slice': 35, 'Squeeze': 7, 'Range': 7, 'Concat': 18, 'Reshape': 41, 'Unsqueeze': 11, 'Sub': 20, 'Mul': 36, 'Add': 37, 'Gather': 3, 'MatMul': 17, 'Split': 3, 'Transpose': 12, 'GreaterOrEqual': 3, 'Softmax': 3, 'GlobalAveragePool': 10, 'Sqrt': 5, 'Reciprocal': 5, 'Erf': 2}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_60...
# {'Shape': 24, 'Cast': 54, 'Slice': 50, 'Squeeze': 9, 'Range': 9, 'Concat': 25, 'Reshape': 59, 'Unsqueeze': 14, 'Sub': 29, 'Mul': 57, 'Add': 56, 'Gather': 3, 'MatMul': 25, 'Split': 5, 'Transpose': 16, 'GreaterOrEqual': 4, 'Softmax': 4, 'GlobalAveragePool': 16, 'Sqrt': 8, 'Reciprocal': 8, 'Erf': 4}
#         FAILED (compilation)                    
# running test mygpt4_trunc_Reshape_80...
# {'Shape': 32, 'Cast': 72, 'Slice': 68, 'Squeeze': 13, 'Range': 13, 'Concat': 33, 'Reshape': 81, 'Unsqueeze': 20, 'Sub': 41, 'Mul': 81, 'Add': 77, 'Gather': 3, 'MatMul': 35, 'Split': 6, 'Transpose': 24, 'GreaterOrEqual': 6, 'Softmax': 6, 'GlobalAveragePool': 22, 'Sqrt': 11, 'Reciprocal': 11, 'Erf': 6}
#         FAILED (compilation)                    
# running test mygpt4_trunc_shape_1...
# {'Shape': 1}ing stage 'setup'...             
#         PASSED                               

# Test Summary:
#         PASSES: 1
#         TOTAL: 11
# results stored in /proj/gdba/shark/chi/src/SHARK-TestSuite/alt_e2eshark/test-run
# #