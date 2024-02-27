module {
  func.func @test_hammingwindow(%arg0: !torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.HammingWindow"(%arg0) : (!torch.vtensor<[],si32>) -> !torch.vtensor<[10],f32>
    return %0 : !torch.vtensor<[10],f32>
  }
}

