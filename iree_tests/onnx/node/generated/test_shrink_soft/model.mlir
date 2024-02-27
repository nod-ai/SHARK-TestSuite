module {
  func.func @test_shrink_soft(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 4 : si64, torch.onnx_meta.opset_version = 9 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Shrink"(%arg0) {torch.onnx.bias = 1.500000e+00 : f32, torch.onnx.lambd = 1.500000e+00 : f32} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32>
    return %0 : !torch.vtensor<[5],f32>
  }
}

