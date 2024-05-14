# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
import torch
from transformers import SegformerImageProcessor, SegformerForImageClassification
from PIL import Image
import requests

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)

# model origin: https://huggingface.co/nvidia/mit-b0
test_modelname = "nvidia/mit-b0"
test_image_url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(test_image_url, stream=True).raw)

processor = SegformerImageProcessor.from_pretrained(test_modelname)
model = SegformerForImageClassification.from_pretrained(test_modelname)

model.to("cpu")
model.eval()

encoding = processor(images=image, return_tensors="pt")

E2ESHARK_CHECK["input"] = encoding["pixel_values"].cpu()
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"]).logits
predicted_class_idx = E2ESHARK_CHECK["output"].argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False

# Post process output to do:
# torch.nn.functional.softmax(output, -1)
# The output logits is the shape of (B, L).
# (batch size, num labels)
# This way we create a probability distribution for each possible label
# when classifying sentence.
E2ESHARK_CHECK["postprocess"] = [
    (torch.nn.functional.softmax, [-1], False, 0),
]
