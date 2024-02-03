import numpy
import onnxruntime


# The generated or checked in onnx file must always be called model.onnx
# the tools/stubs/onnxmodel.py is appended to model.py
# to form runmodel.py in the rundirectory which is then taken
# through flow


# insert here any onnx API call to generate onnx file if
# not using a checked in onnx model

# start an onnxrt session
session = onnxruntime.InferenceSession("model.onnx", None)

# test_input and test_output must be numpy
# bfloat16 is not supported by onnxruntime and numpy
# case to/from fp32 to work around at various stages
# fill the lines that set test_input and onnx_output
# these two are special names and should not be changed
test_input = numpy.random.rand(1, 3, 224, 224).astype(numpy.float32)
print("Input:", test_input)

# Get the name of the input of the model
input_name = session.get_inputs()[0].name
# TBD: Use iobinding and ortvalue explicitly as per
# https://onnxruntime.ai/docs/api/python/api_summary.html
# call inference session
test_output = numpy.array(
    [session.run([], {input_name: test_input})[0]], dtype=numpy.float32
)
print("Output:", test_output)
