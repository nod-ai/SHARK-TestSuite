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
