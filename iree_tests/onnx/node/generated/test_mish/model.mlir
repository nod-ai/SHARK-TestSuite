module {
  func.func public @test_mish(%arg0: !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Mish', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [])"(%arg0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32>
    return %0 : !torch.vtensor<[10000],f32>
  }
  func.func private @"('Mish', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [])"(%arg0: !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = call @"('Softplus', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [])"(%arg0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32>
    %1 = torch.operator "onnx.Tanh"(%0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> 
    %2 = torch.operator "onnx.Mul"(%arg0, %1) : (!torch.vtensor<[10000],f32>, !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> 
    return %2 : !torch.vtensor<[10000],f32>
  }
  func.func private @"('Softplus', '', 18, [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [tensor_type {\0A  elem_type: 1\0A  shape {\0A    dim {\0A      dim_value: 10000\0A    }\0A  }\0A}\0A], [])"(%arg0: !torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Exp"(%arg0) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.CastLike"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[10000],f32>) -> !torch.vtensor<[],f32> 
    %3 = torch.operator "onnx.Add"(%0, %2) : (!torch.vtensor<[10000],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[10000],f32> 
    %4 = torch.operator "onnx.Log"(%3) : (!torch.vtensor<[10000],f32>) -> !torch.vtensor<[10000],f32> 
    return %4 : !torch.vtensor<[10000],f32>
  }
}

