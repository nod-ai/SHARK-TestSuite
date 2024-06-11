module {
  func.func public @test_reduce_log_sum_exp_keepdims_example(%arg0: !torch.vtensor<[3,2,2],f64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('ReduceLogSumExp', '', 18, [tensor_type {\0A  elem_type: 11\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 11\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 1\0Atype: INT\0A])"(%arg0, %arg1) : (!torch.vtensor<[3,2,2],f64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64>
    return %0 : !torch.vtensor<[3,1,2],f64>
  }
  func.func private @"('ReduceLogSumExp', '', 18, [tensor_type {\0A  elem_type: 11\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A, tensor_type {\0A  elem_type: 7\0A  shape {\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 11\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A    dim {\0A      dim_value: 2\0A    }\0A  }\0A}\0A], [name: \22keepdims\22\0Ai: 1\0Atype: INT\0A])"(%arg0: !torch.vtensor<[3,2,2],f64>, %arg1: !torch.vtensor<[1],si64>) -> !torch.vtensor<[3,1,2],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,2,2],f64> 
    %1 = torch.operator "onnx.Exp"(%0) : (!torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,2,2],f64> 
    %2 = torch.operator "onnx.ReduceSum"(%1, %arg1) {torch.onnx.keepdims = 1 : si64} : (!torch.vtensor<[3,2,2],f64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[],f64> 
    %3 = torch.operator "onnx.Log"(%2) : (!torch.vtensor<[],f64>) -> !torch.vtensor<[],f64> 
    %4 = torch.operator "onnx.CastLike"(%3, %arg0) : (!torch.vtensor<[],f64>, !torch.vtensor<[3,2,2],f64>) -> !torch.vtensor<[3,1,2],f64> 
    return %4 : !torch.vtensor<[3,1,2],f64>
  }
}

