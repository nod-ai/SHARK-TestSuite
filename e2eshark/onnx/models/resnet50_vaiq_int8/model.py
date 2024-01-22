import numpy as np
import onnxruntime

# Not ready yet, this is sample code for inference
test_input = np.random.rand(1, 3, 224, 224).astype(np.float32)

# This is printed by run.py with full path
# uncomment to run it locally
# TBD: make it better
# session = onnxruntime.InferenceSession("model.onnx", None)

# Get the name of the input of the model
input_name = session.get_inputs()[0].name
print("Input:", test_input)
outputs = [session.run([], {input_name: test_input})[0]]
print("Output:", outputs)
