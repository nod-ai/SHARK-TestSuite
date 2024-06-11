module {
  func.func public @test_clip_default_inbounds(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Clip', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A, , ], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22\22\0Ainput: \22\22\0Aoutput: \22y\22\0Aop_type: \22Clip\22\0A)"(%arg0, %none, %none) : (!torch.vtensor<[3],f32>, !torch.none, !torch.none) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }
  func.func private @"('Clip', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A, , ], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Ainput: \22\22\0Ainput: \22\22\0Aoutput: \22y\22\0Aop_type: \22Clip\22\0A)"(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.none, %arg2: !torch.none) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    return %0 : !torch.vtensor<[3],f32>
  }
}

