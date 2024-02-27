module {
  func.func @test_castlike_FLOAT_to_FLOAT16_expanded(%arg0: !torch.vtensor<[3,4],f32>, %arg1: !torch.vtensor<[1],f16>) -> !torch.vtensor<[3,4],f16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.saturate = 1 : si64, torch.onnx.to = 10 : si64} : (!torch.vtensor<[3,4],f32>) -> !torch.vtensor<[3,4],f16>
    return %0 : !torch.vtensor<[3,4],f16>
  }
}

