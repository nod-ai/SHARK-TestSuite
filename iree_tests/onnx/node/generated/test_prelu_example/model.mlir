module {
  func.func @test_prelu_example(%arg0: !torch.vtensor<[3,4,5],f32>, %arg1: !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 16 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.PRelu"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],f32>, !torch.vtensor<[3,4,5],f32>) -> !torch.vtensor<[3,4,5],f32>
    return %0 : !torch.vtensor<[3,4,5],f32>
  }
}

