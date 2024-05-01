# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import io
import numpy
import struct
import torch
from typing import Tuple, Optional


def load_torch_save(filename):
    with open(filename, "rb") as f:
        bindata = f.read()
    buf = io.BytesIO(bindata)
    loaded = torch.load(buf)
    return loaded


def get_shape_string(torch_tensor):
    input_shape = list(torch_tensor.shape)
    input_shape_string = "x".join([str(item) for item in input_shape])
    dtype = torch_tensor.dtype
    if dtype == torch.int64:
        input_shape_string += "xi64"
    elif dtype == torch.float32 or dtype == torch.float:
        input_shape_string += "xf32"
    elif dtype == torch.bfloat16 or dtype == torch.float16 or dtype == torch.int16:
        input_shape_string += "xbf16"
    elif dtype == torch.int8:
        input_shape_string += "xi8"
    elif dtype == torch.bool:
        input_shape_string += "xi1"
    else:
        print("In get_shape_string, found an unsupported data type", dtype)
    return input_shape_string


def unpack_bytearray(barray, num_elem, dtype):
    num_array = None
    if dtype == torch.int64:
        num_array = struct.unpack("q" * num_elem, barray)
    elif dtype == torch.float32 or dtype == torch.float:
        num_array = struct.unpack("f" * num_elem, barray)
    elif dtype == torch.bfloat16:
        num_array = struct.unpack("h" * num_elem, barray)
        temptensor = torch.tensor(num_array, dtype=torch.int16)
        rettensor = temptensor.view(dtype=torch.bfloat16)
        return rettensor
    elif dtype == torch.float16:
        num_array = struct.unpack("h" * num_elem, barray)
        temptensor = torch.tensor(num_array, dtype=torch.int16)
        rettensor = temptensor.view(dtype=torch.float16)
        return rettensor
    elif dtype == torch.int32:
        num_array = struct.unpack("l" * num_elem, barray)
    elif dtype == torch.int16:
        num_array = struct.unpack("h" * num_elem, barray)
    elif dtype == torch.int8:
        num_array = struct.unpack("b" * num_elem, barray)
    elif dtype == torch.uint8:
        num_array = struct.unpack("B" * num_elem, barray)
    elif dtype == torch.bool:
        num_array = struct.unpack("?" * num_elem, barray)
    else:
        print("In unpackBytearray, found an unsupported data type", dtype)
    rettensor = torch.tensor(num_array, dtype=dtype)
    return rettensor


def load_raw_binary_as_torch_tensor(binaryfile, shape, dtype):
    # Read the whole files as bytes
    with open(binaryfile, "rb") as f:
        binarydata = f.read()
    # Number of elements in tensor
    num_elem = torch.prod(torch.tensor(list(shape)))
    # Total bytes
    tensor_num_bytes = (num_elem * dtype.itemsize).item()
    barray = bytearray(binarydata[0:tensor_num_bytes])
    # for byte in barray:
    #     print(f"{byte:02X}", end="")
    rettensor = unpack_bytearray(barray, num_elem, dtype)
    reshapedtensor = rettensor.reshape(list(shape))
    f.close()
    return reshapedtensor


def pack_tensor(modelinput):
    mylist = modelinput.flatten().tolist()
    dtype = modelinput.dtype
    if dtype == torch.int64:
        bytearr = struct.pack("%sq" % len(mylist), *mylist)
    elif dtype == torch.float32 or dtype == torch.float:
        bytearr = struct.pack("%sf" % len(mylist), *mylist)
    elif dtype == torch.bfloat16 or dtype == torch.float16:
        reinterprted = modelinput.view(dtype=torch.int16)
        mylist = reinterprted.flatten().tolist()
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.int16:
        bytearr = struct.pack("%sh" % len(mylist), *mylist)
    elif dtype == torch.int8:
        bytearr = struct.pack("%sb" % len(mylist), *mylist)
    elif dtype == torch.uint8:
        bytearr = struct.pack("%sB" % len(mylist), *mylist)
    elif dtype == torch.bool:
        bytearr = struct.pack("%s?" % len(mylist), *mylist)
    else:
        print("In packTensor, found an unsupported data type", dtype)
    return bytearr


def write_inference_input_bin_file(modelinput, modelinputbinfilename):
    with open(modelinputbinfilename, "wb") as f:
        bytearr = pack_tensor(modelinput)
        f.write(bytearr)
        f.close()


class TestTensors:
    """storage class for tuples of tensor-like objects"""

    __slots__ = [
        "data",
        "type",
    ]

    def __init__(self, data: Tuple):
        self.data = data
        self.type = type(self.data[0])
        if not all([type(d) == self.type for d in data]):
            self.type == None

    def __repr__(self):
        return f"TestTensors({self.type}): {self.data.__repr__()}"

    def to_numpy(self) -> "TestTensors":
        """returns a copy of self as a numpy.ndarray type"""
        if self.type == torch.Tensor:
            new_data = tuple([d.numpy() for d in self.data])
            return TestTensors(new_data)
        elif self.type == numpy.ndarray:
            return TestTensors(self.data)
        else:
            raise TypeError(
                f"Unhandled TestTensors conversion from {self.type} to numpy.ndarray"
            )

    def to_torch(self) -> "TestTensors":
        """returns a copy of self as a torch.Tensor type"""
        if self.type == numpy.ndarray:
            new_data = tuple([torch.from_numpy(d) for d in self.data])
            return TestTensors(new_data)
        elif self.type == torch.Tensor:
            return TestTensors(self.data)
        else:
            raise TypeError(
                f"Unhandled TestTensors conversion from {self.type} to torch.Tensor"
            )

    def to_dtype(self, dtype, *, index: Optional[int] = None) -> "TestTensors":
        """returns a copy of self with a converted dtype (at a particular index, if specified)"""
        if self.type == numpy.ndarray:
            if index:
                try:
                    new_data = self.data
                    new_data[index] = new_data[index].astype(dtype)
                except Exception as e:
                    print("to_dtype failed due to excepton {e}.")
            else:
                new_data = tuple([d.astype(dtype) for d in self.data])
        if self.type == torch.Tensor:
            if index:
                try:
                    new_data = self.data
                    new_data[index] = new_data[index].to(dtype=dtype)
                except Exception as e:
                    print("to_dtype failed due to excepton {e}.")
            else:
                new_data = tuple([d.to(dtype=dtype) for d in self.data])
        return TestTensors(new_data)

    def save_to(self, path: str):
        """path should be of the form /path/to/log/folder/unformattedname"""
        if self.type == torch.Tensor:
            data = self.data
        else:
            data = self.to_torch().data
        for i in range(len(data)):
            write_inference_input_bin_file(data[i], path + f".{i}.bin")
