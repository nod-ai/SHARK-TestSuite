module {
  func.func public @test_gelu_tanh_2(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Gelu', '', 20, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22x\22\0Aoutput: \22y\22\0Aop_type: \22Gelu\22\0Aattribute {\0A  name: \22approximate\22\0A  s: \22tanh\22\0A  type: STRING\0A}\0A)"(%arg0) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
  func.func private @"('Gelu', '', 20, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 4\0A    }\0A    dim {\0A      dim_value: 5\0A    }\0A  }\0A}\0A], input: \22x\22\0Aoutput: \22y\22\0Aop_type: \22Gelu\22\0Aattribute {\0A  name: \22approximate\22\0A  s: \22tanh\22\0A  type: STRING\0A}\0A)"(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.636619746> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %5 = torch.operator "onnx.CastLike"(%4, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<4.471500e-02> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %7 = torch.operator "onnx.CastLike"(%6, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %8 = torch.operator "onnx.Sqrt"(%5) : (!torch.vtensor<[],f32>) -> !torch.vtensor<[],f32> 
    %9 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<3.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %10 = torch.operator "onnx.CastLike"(%9, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[],f32> 
    %11 = torch.operator "onnx.Pow"(%arg0, %10) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %12 = torch.operator "onnx.Mul"(%7, %11) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %13 = torch.operator "onnx.Sum"(%arg0, %12) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %14 = torch.operator "onnx.Mul"(%8, %13) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %15 = torch.operator "onnx.Tanh"(%14) : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %16 = torch.operator "onnx.Sum"(%3, %15) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %17 = torch.operator "onnx.Mul"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    %18 = torch.operator "onnx.Mul"(%17, %16) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> 
    return %18 : !torch.vtensor<[3,4,5],f32>
  }
}

