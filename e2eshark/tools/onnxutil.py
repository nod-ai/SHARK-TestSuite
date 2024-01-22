import os, sys, argparse
import onnx


# Given an onxx model, find unique set of ops
def uniqueOnnxOps(model):
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    return ops


# Given an onnx model, return a dictionary of ops with frequency
def frequencyOfOPs(model):
    ops = {}
    for node in model.graph.node:
        if node.op_type in ops:
            ops[node.op_type] += 1
        else:
            ops[node.op_type] = 1
    return ops


if __name__ == "__main__":
    msg = "The script to print contents of an ONNX ProtoBuf file."
    parser = argparse.ArgumentParser(description=msg, epilog="")
    parser.add_argument(
        "inputfile",
        help="Input ONNX file",
    )
    parser.add_argument(
        "-u", "--uniqueops", action="store_true", help="Find unique ops in given file"
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print in human readable format"
    )
    parser.add_argument(
        "-f",
        "--frequency",
        action="store_true",
        help="Print number of occurrences i.e. frquency of ops",
    )

    args = parser.parse_args()
    onnxfile = args.inputfile
    if not os.path.exists(onnxfile):
        print("The given file ", onnxfile, " does not exist\n")
        sys.exit(1)

    model = onnx.load(onnxfile)
    # If it gets past it, model was opened successfully
    print("Successfully opened", onnxfile, "\n")

    if args.print:
        print(model)
    if args.uniqueops:
        ops = uniqueOnnxOps(model)
        print("Number of unique ops:", len(ops), "\nOps: ", ops, "\n")
    if args.frequency:
        ops = frequencyOfOPs(model)
        sortedops = dict(sorted(ops.items(), key=lambda item: item[1], reverse=True))
        count = 0
        for k, v in sortedops.items():
            print(k, ":", v)
            count += v
        print("Total instances of ops: ", count)
