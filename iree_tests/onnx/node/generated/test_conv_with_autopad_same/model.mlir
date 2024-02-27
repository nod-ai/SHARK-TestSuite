module {
  func.func @test_conv_with_autopad_same(%arg0: !torch.vtensor<[1,1,5,5],f32>, %arg1: !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,3,3],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Conv"(%arg0, %arg1) {torch.onnx.auto_pad = "SAME_LOWER", torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,5,5],f32>, !torch.vtensor<[1,1,3,3],f32>) -> !torch.vtensor<[1,1,3,3],f32>
    return %0 : !torch.vtensor<[1,1,3,3],f32>
  }
}

