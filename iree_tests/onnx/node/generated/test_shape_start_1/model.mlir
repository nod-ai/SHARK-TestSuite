module {
  func.func @test_shape_start_1(%arg0: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Shape"(%arg0) {torch.onnx.start = 1 : si64} : (!torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[2],si64>
    return %0 : !torch.vtensor<[2],si64>
  }
}

