# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree import runtime as ireert
import iree.compiler as ireec
from iree.compiler.ir import Context
import torch
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import sys, argparse
import numpy as np

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


class PromptEncoderModule(torch.nn.Module):
    def __init__(
        self,
        hf_model_name,
        precision,
        hf_auth_token=None,
        do_classifier_free_guidance=True,
    ):
        super().__init__()
        self.torch_dtype = torch.float16 if precision == "fp16" else torch.float32
        self.text_encoder_model_1 = CLIPTextModel.from_pretrained(
            hf_model_name,
            subfolder="text_encoder",
            token=hf_auth_token,
        )
        self.text_encoder_model_2 = CLIPTextModelWithProjection.from_pretrained(
            hf_model_name,
            subfolder="text_encoder_2",
            token=hf_auth_token,
        )
        self.do_classifier_free_guidance = do_classifier_free_guidance

    #     self.tokenizer_1 = CLIPTokenizer.from_pretrained(
    #         hf_model_name,
    #         subfolder="tokenizer",
    #         token=hf_auth_token,
    #         model_max_length=max_length,
    #     )
    #     self.tokenizer_2 = CLIPTokenizer.from_pretrained(
    #         hf_model_name,
    #         subfolder="tokenizer_2",
    #         token=hf_auth_token,
    #         model_max_length=max_length,
    #     )
    # def tokenize(self, prompt, negative_prompt):
    #     text_input_ids_1 = self.tokenizer_1(
    #         prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     uncond_input_ids_1 = self.tokenizer_2(
    #         negative_prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     text_input_ids_2 = self.tokenizer_2(
    #         prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     uncond_input_ids_2 = self.tokenizer_2(
    #         negative_prompt,
    #         padding="max_length",
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids
    #     return text_input_ids_1, uncond_input_ids_1, text_input_ids_2, uncond_input_ids_2

    def forward(
        self, text_input_ids_1, text_input_ids_2, uncond_input_ids_1, uncond_input_ids_2
    ):
        with torch.no_grad():
            prompt_embeds_1 = self.text_encoder_model_1(
                text_input_ids_1,
                output_hidden_states=True,
            )
            prompt_embeds_2 = self.text_encoder_model_2(
                text_input_ids_2,
                output_hidden_states=True,
            )
            neg_prompt_embeds_1 = self.text_encoder_model_1(
                uncond_input_ids_1,
                output_hidden_states=True,
            )
            neg_prompt_embeds_2 = self.text_encoder_model_2(
                uncond_input_ids_2,
                output_hidden_states=True,
            )
            # We are only ALWAYS interested in the pooled output of the final text encoder
            pooled_prompt_embeds = prompt_embeds_2[0]
            neg_pooled_prompt_embeds = neg_prompt_embeds_2[0]

            prompt_embeds_list = [
                prompt_embeds_1.hidden_states[-2],
                prompt_embeds_2.hidden_states[-2],
            ]
            neg_prompt_embeds_list = [
                neg_prompt_embeds_1.hidden_states[-2],
                neg_prompt_embeds_2.hidden_states[-2],
            ]

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
            neg_prompt_embeds = torch.concat(neg_prompt_embeds_list, dim=-1)

            bs_embed, seq_len, _ = prompt_embeds.shape

            prompt_embeds = prompt_embeds.repeat(1, 1, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * 1, seq_len, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1).view(
                bs_embed * 1, -1
            )
            add_text_embeds = pooled_prompt_embeds
            if self.do_classifier_free_guidance:
                neg_pooled_prompt_embeds = neg_pooled_prompt_embeds.repeat(1, 1).view(
                    1, -1
                )
                neg_prompt_embeds = neg_prompt_embeds.repeat(1, 1, 1)
                neg_prompt_embeds = neg_prompt_embeds.view(bs_embed * 1, seq_len, -1)
                prompt_embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
                add_text_embeds = torch.cat(
                    [neg_pooled_prompt_embeds, add_text_embeds], dim=0
                )

            add_text_embeds = add_text_embeds.to(self.torch_dtype)
            prompt_embeds = prompt_embeds.to(self.torch_dtype)
            return prompt_embeds, add_text_embeds


# def export_prompt_encoder(
#     hf_model_name,
#     hf_auth_token=None,
#     max_length=64,
#     precision="fp16",
#     compile_to="torch",
#     external_weights=None,
#     external_weight_path=None,
#     device=None,
#     target_triple=None,
#     ireec_flags=None,
#     exit_on_vmfb=True,
#     pipeline_dir=None,
#     input_mlir=None,
#     attn_spec=None,
#     weights_only=False,
# ):
#     if "turbo" in hf_model_name:
#         do_classifier_free_guidance = False
#     else:
#         do_classifier_free_guidance = True

#     if attn_spec in ["default", "", None]:
#         attn_spec = os.path.join(
#             os.path.realpath(os.path.dirname(__file__)), "default_mfma_attn_spec.mlir"
#         )

#     if pipeline_dir not in [None, ""]:
#         safe_name = os.path.join(pipeline_dir, "prompt_encoder")
#     else:
#         safe_name = utils.create_safe_name(
#             hf_model_name, f"-{str(max_length)}-{precision}-prompt-encoder-{device}"
#         )
#     if input_mlir:
#         vmfb_path = utils.compile_to_vmfb(
#             input_mlir,
#             device,
#             target_triple,
#             ireec_flags,
#             safe_name,
#             mlir_source="file",
#             return_path=not exit_on_vmfb,
#             const_expr_hoisting=True,
#             attn_spec=attn_spec,
#         )
#         return vmfb_path
#     # Load the tokenizer and text encoder to tokenize and encode the text.
#     tokenizer_1 = CLIPTokenizer.from_pretrained(
#         hf_model_name,
#         subfolder="tokenizer",
#         token=hf_auth_token,
#         model_max_length=max_length,
#     )
#     tokenizer_2 = CLIPTokenizer.from_pretrained(
#         hf_model_name,
#         subfolder="tokenizer_2",
#         token=hf_auth_token,
#         model_max_length=max_length,
#     )
#     tokenizers = [tokenizer_1, tokenizer_2]
#     prompt_encoder_module = PromptEncoderModule(
#         hf_model_name, precision, hf_auth_token, do_classifier_free_guidance
#     )
#     if precision == "fp16":
#         prompt_encoder_module = prompt_encoder_module.half()
#     mapper = {}

#     utils.save_external_weights(
#         mapper, prompt_encoder_module, external_weights, external_weight_path
#     )

#     if weights_only:
#         return external_weight_path

#     class CompiledClip(CompiledModule):
#         if external_weights:
#             params = export_parameters(
#                 prompt_encoder_module,
#                 external=True,
#                 external_scope="",
#                 name_mapper=mapper.get,
#             )
#         else:
#             params = export_parameters(prompt_encoder_module)

#         def encode_prompts(
#             self,
#             t_ids_1=AbstractTensor(1, max_length, dtype=torch.int64),
#             t_ids_2=AbstractTensor(1, max_length, dtype=torch.int64),
#             uc_ids_1=AbstractTensor(1, max_length, dtype=torch.int64),
#             uc_ids_2=AbstractTensor(1, max_length, dtype=torch.int64),
#         ):
#             return jittable(prompt_encoder_module.forward)(
#                 t_ids_1, t_ids_2, uc_ids_1, uc_ids_2
#             )

#     import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
#     inst = CompiledClip(context=Context(), import_to=import_to)

#     module_str = str(CompiledModule.get_mlir_module(inst))

#     if compile_to != "vmfb":
#         return module_str, tokenizers
#     else:
#         vmfb_path = utils.compile_to_vmfb(
#             module_str,
#             device,
#             target_triple,
#             ireec_flags,
#             safe_name,
#             return_path=not exit_on_vmfb,
#             const_expr_hoisting=True,
#             attn_spec=attn_spec,
#         )
#         return module_str, vmfb_path


# if __name__ == "__main__":
#     from turbine_models.custom_models.sdxl_inference.sdxl_cmd_opts import args

#     mod_str, _ = export_prompt_encoder(
#         args.hf_model_name,
#         args.hf_auth_token,
#         args.max_length,
#         args.precision,
#         args.compile_to,
#         args.external_weights,
#         args.external_weight_path,
#         args.device,
#         args.iree_target_triple,
#         args.ireec_flags + args.clip_flags,
#         exit_on_vmfb=True,
#         pipeline_dir=args.pipeline_dir,
#         input_mlir=args.input_mlir,
#         attn_spec=args.attn_spec,
#     )
#     if args.input_mlir:
#         exit()
#     safe_name_1 = safe_name = utils.create_safe_name(
#         args.hf_model_name, f"_{str(args.max_length)}_{args.precision}_prompt_encoder"
#     )
#     with open(f"{safe_name}.mlir", "w+") as f:
#         f.write(mod_str)
#     print("Saved to", safe_name + ".mlir")
        
model = PromptEncoderModule("stabilityai/stable-diffusion-xl-base-1.0", "fp32")

tokenizer_1 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer",
)
tokenizer_2 = CLIPTokenizer.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="tokenizer_2",
)
text_input_ids_list = []
uncond_input_ids_list = []

# Tokenize prompt and negative prompt.
tokenizers = [tokenizer_1, tokenizer_2]
for tokenizer in tokenizers:
    text_inputs = tokenizer(
        "A very fast car leaving a trail of fire as it screams along a mountain road, old school racing animation, retro 1980s anime style, 4k, motion blur, action shot, semi-realistic, nightwave, neon, tokyo",
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    uncond_input = tokenizer(
        "Watermark, blurry, oversaturated, low resolution, pollution",
        padding="max_length",
        max_length=64,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    uncond_input_ids = uncond_input.input_ids

    text_input_ids_list.extend([text_input_ids])
    uncond_input_ids_list.extend([uncond_input_ids])

E2ESHARK_CHECK["input"] = [text_input_ids_list[0], text_input_ids_list[1], uncond_input_ids_list[0], uncond_input_ids_list[1]]

E2ESHARK_CHECK["output"] = model.forward(text_input_ids_list[0], text_input_ids_list[1], uncond_input_ids_list[0], uncond_input_ids_list[1])

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])