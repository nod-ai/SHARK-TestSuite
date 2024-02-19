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
    # Apply listed function (tools/stub and run.pl must be able to find definition)
    # on output from target in sequence to post process output and compare the final
    # output instead.
    # First arg to function is the output, any additional args should be added as a list
    # list of function and its 1+args as tuple should be provided
    # Exmaple: "postprocess": [(torch.nn.functional.softmax, [0])]
    # which will be called as output = torch.nn.functional.softmax(output, 0)
    "postprocess": None,
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
def postProcess(E2ESHARK_CHECK):
    test_output = E2ESHARK_CHECK["output"]
    postprocess_output = []
    # Call chain of post processing -- run.pl will do same on backend inference output
    if E2ESHARK_CHECK.get("postprocess"):
        for item in test_output:
            # Run post processing pipeline
            for func, argextra in E2ESHARK_CHECK["postprocess"]:
                item = func(item, *argextra)
            postprocess_output += [item]
    else:
        postprocess_output = test_output
    return postprocess_output
