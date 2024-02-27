module {
  func.func @test_stft_with_window(%arg0: !torch.vtensor<[1,128,1],f32>, %arg1: !torch.vtensor<[],si64>, %arg2: !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,15,9,2],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.STFT"(%arg0, %arg1, %arg2) : (!torch.vtensor<[1,128,1],f32>, !torch.vtensor<[],si64>, !torch.vtensor<[16],f32>) -> !torch.vtensor<[1,15,9,2],f32>
    return %0 : !torch.vtensor<[1,15,9,2],f32>
  }
}

