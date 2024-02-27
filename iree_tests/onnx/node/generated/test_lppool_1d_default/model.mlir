module {
  func.func @test_lppool_1d_default(%arg0: !torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.LpPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64], torch.onnx.p = 3 : si64, torch.onnx.strides = [1 : si64]} : (!torch.vtensor<[1,3,32],f32>) -> !torch.vtensor<[1,3,31],f32>
    return %0 : !torch.vtensor<[1,3,31],f32>
  }
}

