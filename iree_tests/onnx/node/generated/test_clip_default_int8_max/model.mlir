module {
  func.func public @test_clip_default_int8_max(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Clip', '', 17, [tensor_type {\0A  elem_type: 3\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, , tensor_type {\0A  elem_type: 3\0A  shape {\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 3\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22\22\0Ainput: \22max\22\0Aoutput: \22y\22\0Aop_type: \22Clip\22\0A)"(%arg0, %none, %arg1) : (!torch.vtensor<[3,4,5],si8>, !torch.none, !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8>
    return %0 : !torch.vtensor<[3,4,5],si8>
  }
  func.func private @"('Clip', '', 17, [tensor_type {\0A  elem_type: 3\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, , tensor_type {\0A  elem_type: 3\0A  shape {\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 3\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22\22\0Ainput: \22max\22\0Aoutput: \22y\22\0Aop_type: \22Clip\22\0A)"(%arg0: !torch.vtensor<[3,4,5],si8>, %arg1: !torch.none, %arg2: !torch.vtensor<[],si8>) -> !torch.vtensor<[3,4,5],si8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Less"(%arg2, %arg0) : (!torch.vtensor<[],si8>, !torch.vtensor<[3,4,5],si8>) -> !torch.vtensor<[3,4,5],i1> 
    %1 = torch.operator "onnx.Where"(%0, %arg2, %arg0) : (!torch.vtensor<[3,4,5],i1>, !torch.vtensor<[],si8>, !torch.vtensor<[3,4,5],si8>) -> !torch.vtensor<[3,4,5],si8> 
    return %1 : !torch.vtensor<[3,4,5],si8>
  }
}

