module {
  func.func @test_averagepool_3d_dilations_large_count_include_pad_is_1_ceil_mode_is_False(%arg0: !torch.vtensor<[1,1,32,32,32],f32>) -> !torch.vtensor<[1,1,8,8,8],f32> attributes {torch.onnx_meta.ir_version = 9 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.count_include_pad = 1 : si64, torch.onnx.dilations = [2 : si64, 2 : si64, 2 : si64], torch.onnx.kernel_shape = [5 : si64, 5 : si64, 5 : si64], torch.onnx.strides = [3 : si64, 3 : si64, 3 : si64]} : (!torch.vtensor<[1,1,32,32,32],f32>) -> !torch.vtensor<[1,1,8,8,8],f32>
    return %0 : !torch.vtensor<[1,1,8,8,8],f32>
  }
}

