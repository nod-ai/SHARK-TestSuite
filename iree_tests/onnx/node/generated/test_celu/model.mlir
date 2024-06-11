module {
  func.func public @test_celu(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Celu', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], input: \22X\22\0Aoutput: \22Y\22\0Aop_type: \22Celu\22\0Aattribute {\0A  name: \22alpha\22\0A  f: 2\0A  type: FLOAT\0A}\0A)"(%arg0) : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32>
    return %0 : !torch.vtensor<[3,3,3,1],f32>
  }
  func.func private @"('Celu', '', 17, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 3\0A    }\0A    dim {\0A      dim_value: 1\0A    }\0A  }\0A}\0A], input: \22X\22\0Aoutput: \22Y\22\0Aop_type: \22Celu\22\0Aattribute {\0A  name: \22alpha\22\0A  f: 2\0A  type: FLOAT\0A}\0A)"(%arg0: !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32> 
    %1 = torch.operator "onnx.Div"(%arg0, %0) : (!torch.vtensor<[3,3,3,1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %2 = torch.operator "onnx.Elu"(%1) {torch.onnx.alpha = 1.000000e+00 : f32} : (!torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    %3 = torch.operator "onnx.Mul"(%0, %2) : (!torch.vtensor<[1],f32>, !torch.vtensor<[3,3,3,1],f32>) -> !torch.vtensor<[3,3,3,1],f32> 
    return %3 : !torch.vtensor<[3,3,3,1],f32>
  }
}

