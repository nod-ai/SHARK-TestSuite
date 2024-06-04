module {
  func.func public @test_reduce_l2_empty_set(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('ReduceL2', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 0\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 1\0Atype: INT\0A])"(%arg0, %arg1) : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32>
    return %0 : !torch.vtensor<[2,1,4],f32>
  }
  func.func private @"('ReduceL2', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 0\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 1\0Atype: INT\0A])"(%arg0: !torch.vtensor<[2,0,4],f32>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[2,1,4],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mul"(%arg0, %arg0) : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[2,0,4],f32>) -> !torch.vtensor<[2,0,4],f32> 
    %1 = torch.operator "onnx.ReduceSum"(%0, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[2,0,4],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Cast"(%1) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Sqrt"(%2) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.CastLike"(%3, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[2,0,4],f32>) -> !torch.vtensor<[2,1,4],f32> 
    return %4 : !torch.vtensor<[2,1,4],f32>
  }
}

