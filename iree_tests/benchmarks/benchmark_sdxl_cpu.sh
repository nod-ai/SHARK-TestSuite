#!/bin/bash
#
# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -xeuo pipefail

this_dir="$(cd $(dirname $0) && pwd)"
repo_root="$(cd $this_dir/../.. && pwd)"
vae_decode_dir="$repo_root//pytorch/models/sdxl-vae-decode-tank"
scheduled_unet_dir="$repo_root/pytorch/models/sdxl-scheduled-unet-3-tank"
prompt_encoder_dir="$repo_root/pytorch/models/sdxl-prompt-encoder-tank"

echo "Running sdxl benchmark"

iree-benchmark-module --device=local-task --module="$prompt_encoder_dir/model_sdxl_cpu_llvm_task_real_weights.vmfb" --parameters=model="$prompt_encoder_dir/real_weights.irpa" --module="$scheduled_unet_dir/model_sdxl_cpu_llvm_task_real_weights.vmfb" --parameters=model="$scheduled_unet_dir/real_weights.irpa" --module="$vae_decode_dir/model_sdxl_cpu_llvm_task_real_weights.vmfb" --parameters=model="$vae_decode_dir/real_weights.irpa" --module="$this_dir/sdxl_full_pipeline_fp16_.vmfb" --function=tokens_to_image --input=1x4x128x128xf16 --input=1xf16 --input=1x64xi64 --input=1x64xi64 --input=1x64xi64 --input=1x64xi64

echo "Succesfully finished sdxl pipeline benchmark"