module {
  func.func @test_thresholdedrelu_example(%arg0: !torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32> attributes {torch.onnx_meta.ir_version = 5 : si64, torch.onnx_meta.opset_version = 10 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.ThresholdedRelu"(%arg0) {torch.onnx.alpha = 2.000000e+00 : f32} : (!torch.vtensor<[5],f32>) -> !torch.vtensor<[5],f32>
    return %0 : !torch.vtensor<[5],f32>
  }
}

