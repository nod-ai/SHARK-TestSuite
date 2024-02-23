# Copyright 2024 Advanced Micro Devices
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# These are pickle-saved and used by tools/stubs python and run.pl.
# If adding new fields, make sure the field has default value and have updated
# tools/stubs and run.pl to handle the new fields
E2ESHARK_CHECK_DEF = {
    # this is input applied to the model
    "input": None,
    # this is output gotten from the model
    "output": None,
    # Controls how to import a graph from PyTorch into MLIR, options are: compile or fximport
    "torchmlirimport": "fximport",
    # By default, the input.to(dtype) is called, set it to False to not do so
    "inputtodtype": True,
    # Apply listed function with its arguments and return value selection repitively
    # to post-process an output. Each entry in list should be a tuple with four
    # entries: (function, [args other than input], isReturnTuple, indexOfTupleForTupleReturn)
    # Exmaple: "postprocess": [(torch.nn.functional.softmax, [0], False, 0), (torch.topk, [5], True, 1)]
    # which will be called as output = topk(torch.nn.functional.softmax(output, 0), 5)[1]
    "postprocess": None,
    # Store the post-processed output here
    "postprocessed_output": None,
}


# fix model op to be a list of tensors
# recursively flattens tuple of tuples to a list of single tuples in order
# example - (1, ((2,3),(4,5),(6,7))) -> [1,2,3,4,5,6,7]
def getOutputTensorList(test_out):
    def flatten_tuples(tup):
        if isinstance(tup, tuple):
            res = []
            for t in tup:
                res.extend(flatten_tuples(t))
            return res
        return [tup]

    test_op_list = flatten_tuples(test_out)
    return test_op_list


# Apply post processing functions on output
def postProcess(e2esharkDict):
    test_output = e2esharkDict["output"]
    postprocess_output = []
    # Call chain of post processing -- run.pl will do same on backend inference output
    if e2esharkDict.get("postprocess"):
        for item in test_output:
            # Run post processing pipeline
            for func, argextra, isRetTuple, tupleIndex in e2esharkDict["postprocess"]:
                if len(argextra) > 0:
                    item = func(item, *argextra)
                else:
                    item = func(item)
                if isRetTuple:
                    item = item[tupleIndex]
            postprocess_output += [item]
    else:
        postprocess_output = test_output
    return postprocess_output
