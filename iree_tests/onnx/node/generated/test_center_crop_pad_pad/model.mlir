module {
  func.func public @test_center_crop_pad_pad(%arg0: !torch.vtensor<[10,7,3],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('CenterCropPad', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10\0A    }\0A    dim {\0A      dim_value: 7\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 20\0A    }\0A    dim {\0A      dim_value: 10\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22shape\22\0Aoutput: \22y\22\0Aop_type: \22CenterCropPad\22\0A)"(%arg0, %arg1) : (!torch.vtensor<[10,7,3],f32>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,3],f32>
    return %0 : !torch.vtensor<[20,10,3],f32>
  }
  func.func private @"('CenterCropPad', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10\0A    }\0A    dim {\0A      dim_value: 7\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 20\0A    }\0A    dim {\0A      dim_value: 10\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22shape\22\0Aoutput: \22y\22\0Aop_type: \22CenterCropPad\22\0A)"(%arg0: !torch.vtensor<[10,7,3],f32>, %arg1: !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %1 = torch.operator "onnx.Shape"(%arg0) : (!torch.vtensor<[10,7,3],f32>) -> !torch.vtensor<[3],si64> 
    %2 = torch.operator "onnx.Max"(%1, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %3 = torch.operator "onnx.Sub"(%2, %1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %4 = torch.operator "onnx.Div"(%3, %0) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    %5 = torch.operator "onnx.Sub"(%3, %4) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %6 = torch.operator "onnx.Concat"(%4, %5) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[6],si64> 
    %7 = torch.operator "onnx.Pad"(%arg0, %6) : (!torch.vtensor<[10,7,3],f32>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[?,?,?],f32> 
    %8 = torch.operator "onnx.Shape"(%7) : (!torch.vtensor<[?,?,?],f32>) -> !torch.vtensor<[3],si64> 
    %9 = torch.operator "onnx.Sub"(%8, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %10 = torch.operator "onnx.Div"(%9, %0) : (!torch.vtensor<[3],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3],si64> 
    %11 = torch.operator "onnx.Add"(%10, %arg1) : (!torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[3],si64> 
    %12 = torch.operator "onnx.Slice"(%7, %10, %11) : (!torch.vtensor<[?,?,?],f32>, !torch.vtensor<[3],si64>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[20,10,3],f32> 
    return %12 : !torch.vtensor<[20,10,3],f32>
  }
}

