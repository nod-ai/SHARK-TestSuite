module {
  func.func public @test_gelu_default_1(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Gelu', '', 20, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Aoutput: \22y\22\0Aop_type: \22Gelu\22\0A)"(%arg0) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }
  func.func private @"('Gelu', '', 20, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A  }\0A}\0A], input: \22x\22\0Aoutput: \22y\22\0Aop_type: \22Gelu\22\0A)"(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.CastLike"(%4, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Sqrt"(%5) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %7 = torch.operator "onnx.Div"(%arg0, %6) : (!torch.vtensor<[3],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3],f32> 
    %8 = torch.operator "onnx.Erf"(%7) : (!torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %9 = torch.operator "onnx.Sum"(%3, %8) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %10 = torch.operator "onnx.Mul"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    %11 = torch.operator "onnx.Mul"(%10, %9) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> 
    return %11 : !torch.vtensor<[3],f32>
  }
}

