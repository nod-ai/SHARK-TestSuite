import io, torch, struct
from typing import Tuple, Optional
import numpy


def loadTorchSave(filename):
    with open(filename, "rb") as f:
        bindata = f.read()
    buf = io.BytesIO(bindata)
    loaded = torch.load(buf)
    return loaded


def getShapeString(torchtensor):
    inputshape = list(torchtensor.shape)
    inputshapestring = "x".join([str(item) for item in inputshape])
    dtype = torchtensor.dtype
    if dtype == torch.int64:
        inputshapestring += "xi64"
    elif dtype == torch.float32 or dtype == torch.float:
        inputshapestring += "xf32"
    elif dtype == torch.bfloat16 or dtype == torch.float16 or dtype == torch.int16:
        inputshapestring += "xbf16"
    elif dtype == torch.int8:
        inputshapestring += "xi8"
    elif dtype == torch.bool:
        inputshapestring += "xi1"
    else:
        print("In getShapeString, found an unsupported data type", dtype)
    return inputshapestring


def unpackBytearray(barray, num_elem, dtype):
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


def loadRawBinaryAsTorchSensor(binaryfile, shape, dtype):
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
    rettensor = unpackBytearray(barray, num_elem, dtype)
    reshapedtensor = rettensor.reshape(list(shape))
    f.close()
    return reshapedtensor


def packTensor(modelinput):
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
    elif dtype == torch.bool:
        bytearr = struct.pack("%s?" % len(mylist), *mylist)
    else:
        print("In packTensor, found an unsupported data type", dtype)
    return bytearr


def writeInferenceInputBinFile(modelinput, modelinputbinfilename):
    with open(modelinputbinfilename, "wb") as f:
        bytearr = packTensor(modelinput)
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
        if self.type == torch.Tensor:
            for d in self.data:
                writeInferenceInputBinFile(d, path)
        else:
            for d in self.to_torch().data:
                writeInferenceInputBinFile(d, path)
