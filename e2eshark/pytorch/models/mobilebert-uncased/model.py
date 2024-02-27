import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import MobileBertModel, AutoTokenizer

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = dict(E2ESHARK_CHECK_DEF)


test_modelname = "google/mobilebert-uncased"
tokenizer = AutoTokenizer.from_pretrained(test_modelname)
model = MobileBertModel.from_pretrained(
    test_modelname,
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False,
    torchscript=True,
)
model.config.pad_token_id = None
model.to("cpu")
model.eval()
E2ESHARK_CHECK["input"] = torch.randint(2, (1, 128))
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])

print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
# For geneartive AI models, input is int and should be kept that way for
# casted models as well
E2ESHARK_CHECK["inputtodtype"] = False
