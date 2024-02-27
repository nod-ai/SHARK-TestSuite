module {
  func.func @test_cast_FLOAT_to_DOUBLE(%arg0: !torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 11 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f64>
    return %0 : !torch.vtensor<[3,4],f64>
  }
}

