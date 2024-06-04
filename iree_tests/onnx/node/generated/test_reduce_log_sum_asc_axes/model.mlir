module {
  func.func public @test_reduce_log_sum_asc_axes(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('ReduceLogSum', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 0\0Atype: INT\0A])"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[5],f32>
    return %0 : !torch.vtensor<[5],f32>
  }
  func.func private @"('ReduceLogSum', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 0\0Atype: INT\0A])"(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[2],si64>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.ReduceSum"(%arg0, %arg1) {torch.onnx.keepdims = 0 : si64} : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.Log"(%0) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[5],f32> 
    return %1 : !torch.vtensor<[5],f32>
  }
}

