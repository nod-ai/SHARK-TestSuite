module {
  func.func @test_maxpool_2d_dilations(%arg0: !torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.dilations = [2 : si64, 2 : si64], torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,1,4,4],f32>) -> !torch.vtensor<[1,1,2,2],f32>
    return %0 : !torch.vtensor<[1,1,2,2],f32>
  }
}

