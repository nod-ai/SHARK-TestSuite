import numpy as np
import onnxruntime


# This file constitutes end part of runmodel.py
# tools/stubs/onnxstartmodel.py + model.py in test dir + this file
# makes up the runmodel.py

# input_name and onnx_output will be declared and set
# by middle model.py

print("Input:", test_input)
np.save(inputsavefilename, test_input)

print("Onput:", onnx_output)
np.save(outputsavefilename, onnx_output)
