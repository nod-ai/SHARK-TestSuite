module {
  func.func @test_leakyrelu_example_expanded(%arg0: !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value_float = 1.000000e-01 : f32} : () -> !torch.vtensor<[],f32>
    %1 = torch.operator "onnx.CastLike"(%0, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32>
    %2 = torch.vtensor.literal(dense<0.000000e+00> : tensor<f32>) : !torch.vtensor<[],f32>
    %3 = torch.operator "onnx.CastLike"(%2, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[],f32>
    %4 = torch.operator "onnx.Less"(%arg0, %3) : (!torch.vtensor<[3],f32>, !torch.vtensor<[],f32>) -> !torch.vtensor<[3],i1>
    %5 = torch.operator "onnx.Mul"(%1, %arg0) : (!torch.vtensor<[],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    %6 = torch.operator "onnx.Where"(%4, %5, %arg0) : (!torch.vtensor<[3],i1>, !torch.vtensor<[3],f32>, !torch.vtensor<[3],f32>) -> !torch.vtensor<[3],f32>
    return %6 : !torch.vtensor<[3],f32>
  }
}

