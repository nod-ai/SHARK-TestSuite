module {
  func.func @test_identity(%arg0: !torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Identity"(%arg0) : (!torch.vtensor<[1,1,2,2],f32>) -> !torch.vtensor<[1,1,2,2],f32>
    return %0 : !torch.vtensor<[1,1,2,2],f32>
  }
}

