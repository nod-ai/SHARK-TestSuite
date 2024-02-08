import numpy
import onnxruntime
import torch


# The generated or checked in onnx file must always be called model.onnx
# the tools/stubs/onnxmodel.py is appended to model.py
# to form runmodel.py in the rundirectory which is then taken
# through flow


# start an onnxrt session
session = onnxruntime.InferenceSession("model.onnx", None)

# Even if model is quantized, the inputs and outputs are
# not, so apply float32
test_input_X = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)

# gets X in inputs[0] and Y in inputs[1]
inputs = session.get_inputs()
# gets Z in outputs[0]
outputs = session.get_outputs()

# test_input and test_output are list of numpy arrays
# each index into list is one input or one output in the
# order it appears in the model
test_input = [test_input_X]

test_output = session.run(
    [outputs[0].name],
    {inputs[0].name: test_input_X},
)

print("Input:", test_input)
print("Output:", test_output)
