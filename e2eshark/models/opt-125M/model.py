import sys, argparse
import torch
import torch.nn as nn
import torch_mlir
from transformers import OPTModel, AutoTokenizer

test_model_name = "facebook/opt-125M"


class model_opt_125M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = OPTModel.from_pretrained(
            test_model_name,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, tokens):
        return self.model.forward(tokens)[0]

    def name(self):
        return self.__class__.__name__


if __name__ == "__main__":
    msg = "The script to run a Pytorch model test"
    runstages = ["pytorch", "onnx", "torchmlir"]
    parser = argparse.ArgumentParser(description=msg, epilog="")

    parser.add_argument(
        "-d",
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Tensor datatype to use",
    )
    parser.add_argument(
        "-r",
        "--runto",
        choices=runstages,
        default=runstages[len(runstages) - 1],
        help="Run up to a particular stage of testing",
    )
    args = parser.parse_args()
    dtype = args.dtype
    runto = args.runto

    # Instantiate model and run inference in PyTorch
    runstage = runstages[0]
    model = model_opt_125M()
    model_name_dtype = model.name() + "." + dtype
    # batch size 8 of input size 3
    tokenizer = AutoTokenizer.from_pretrained(test_model_name)
    test_input = torch.tensor([tokenizer.encode("The test prommpt")])

    if dtype == "bf16":
        model = model.to(torch.bfloat16)
        test_input = test_input.to(torch.bfloat16)

    output_pytorch = model(test_input)
    print("Pytorch output: ", output_pytorch)
    if runto == runstages[0]:
        sys.exit(0)

    # runstage ONNX
    onnx_name = model_name_dtype + "." + runstages[1]
    onnx_program = torch.onnx.export(model, test_input, onnx_name)

    if runto == runstages[1]:
        sys.exit(0)

        # runstage torch MLIR
    torch_mlir_name = model_name_dtype + "." + runstages[2]
    ts_model = torch.jit.script(model)
    torch_mlir_model = torch_mlir.compile(
        ts_model,
        (test_input),
        output_type="torch",
        use_tracing=True,
        verbose=False,
    )
    with open(torch_mlir_name, "w+") as f:
        f.write(torch_mlir_model.operation.get_asm())
