import sys, argparse
import torch_mlir

if __name__ == "__main__":
    msg = "The script to run a model test"
    parser = argparse.ArgumentParser(description=msg, epilog="")

    parser.add_argument(
        "-d",
        "--dtype",
        choices=["fp32", "bf16"],
        default="fp32",
        help="Tensor datatype to use",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["torch", "onnx", "ort"],
        default="onnx",
        help="Generate torch MLIR, ONNX or ONNX plus ONNX RT stub",
    )
    parser.add_argument(
        "-o",
        "--outfileprefix",
        help="Prefix of output files written by this model",
    )
    args = parser.parse_args()
    dtype = args.dtype
    runmode = args.mode
    outfileprefix = args.outfileprefix

    if not outfileprefix:
        outfileprefix = model.name()

    outfileprefix += "." + dtype

    if dtype == "bf16":
        model = model.to(torch.bfloat16)
        test_input = test_input.to(torch.bfloat16)

    inputsavefilename = outfileprefix + ".input.pt"
    print("Input:", test_input)
    torch.save(test_input, inputsavefilename)

    output_pytorch = model(test_input)

    outputsavefilename = outfileprefix + ".output.pt"
    torch.save(output_pytorch, outputsavefilename)
    print("Pytorch output:", output_pytorch)

    if runmode == "onnx" or runmode == "ort":
        onnx_name = outfileprefix + ".onnx"
        onnx_program = torch.onnx.export(model, test_input, onnx_name)
    elif runmode == "torchmlir":
        torch_mlir_name = outfileprefix + ".pytorch.torch.mlir"
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
