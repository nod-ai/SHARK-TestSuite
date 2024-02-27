module {
  func.func @test_bernoulli_expanded(%arg0: !torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.RandomUniformLike"(%arg0) {torch.onnx.dtype = 11 : si64, torch.onnx.high = 1.000000e+00 : f32, torch.onnx.low = 0.000000e+00 : f32} : (!torch.vtensor<[10],f64>) -> !torch.vtensor<[10],f64>
    %1 = torch.operator "onnx.Greater"(%0, %arg0) : (!torch.vtensor<[10],f64>, !torch.vtensor<[10],f64>) -> !torch.vtensor<[10],i1>
    %2 = torch.operator "onnx.Cast"(%1) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[10],i1>) -> !torch.vtensor<[10],f64>
    return %2 : !torch.vtensor<[10],f64>
  }
}

