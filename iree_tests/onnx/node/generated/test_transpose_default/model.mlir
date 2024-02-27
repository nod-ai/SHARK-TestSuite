module {
  func.func @test_transpose_default(%arg0: !torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Transpose"(%arg0) : (!torch.vtensor<[2,3,4],f32>) -> !torch.vtensor<[4,3,2],f32>
    return %0 : !torch.vtensor<[4,3,2],f32>
  }
}

