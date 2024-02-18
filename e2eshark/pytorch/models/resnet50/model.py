import sys, os
import torch
import torch.nn as nn
import torch_mlir
from torchvision.models import resnet50, ResNet50_Weights

# import from e2eshark/tools to allow running in current dir, for run through
# run.pl, commutils is symbolically linked to allow any rundir to work
sys.path.insert(0, "../../../tools/stubs")
from commonutils import E2ESHARK_CHECK_DEF

# Create an instance of it for this test
E2ESHARK_CHECK = E2ESHARK_CHECK_DEF

test_modelname = "resnet50"
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

E2ESHARK_CHECK["input"] = E2ESHARK_CHECK["input"] = torch.randn(1, 3, 224, 224)
E2ESHARK_CHECK["output"] = model(E2ESHARK_CHECK["input"])
print("Input:", E2ESHARK_CHECK["input"])
print("Output:", E2ESHARK_CHECK["output"])
