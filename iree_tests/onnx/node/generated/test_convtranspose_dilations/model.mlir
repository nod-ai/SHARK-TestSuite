module {
  func.func @test_convtranspose_dilations(%arg0: !torch.vtensor<[1,1,3,3],f32>, %arg1: !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,5,5],f32> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.ConvTranspose"(%arg0, %arg1) {torch.onnx.dilations = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,1,3,3],f32>, !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,5,5],f32>
    return %0 : !torch.vtensor<[1,1,5,5],f32>
  }
}

