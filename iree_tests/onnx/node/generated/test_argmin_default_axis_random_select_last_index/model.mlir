module {
  func.func @test_argmin_default_axis_random_select_last_index(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,3,4],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.ArgMin"(%arg0) {torch.onnx.keepdims = 1 : si64, torch.onnx.select_last_index = 1 : si64} : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[1,3,4],si64>
    return %0 : !torch.vtensor<[1,3,4],si64>
  }
}

