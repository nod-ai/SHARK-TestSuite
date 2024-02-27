module {
  func.func @test_bernoulli_double(%arg0: !torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Bernoulli"(%arg0) {torch.onnx.dtype = 11 : si64} : (!torch.vtensor<[10],f32>) -> !torch.vtensor<[10],f64>
    return %0 : !torch.vtensor<[10],f64>
  }
}

