module {
  func.func @test_maxpool_3d_default(%arg0: !torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 12 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,32,32,32],f32>) -> !torch.vtensor<[1,3,31,31,31],f32>
    return %0 : !torch.vtensor<[1,3,31,31,31],f32>
  }
}

