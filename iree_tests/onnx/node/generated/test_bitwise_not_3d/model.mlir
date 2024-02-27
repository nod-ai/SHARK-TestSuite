module {
  func.func @test_bitwise_not_3d(%arg0: !torch.vtensor<[3,4,5],ui16>) -> !torch.vtensor<[3,4,5],ui16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.BitwiseNot"(%arg0) : (!torch.vtensor<[3,4,5],ui16>) -> !torch.vtensor<[3,4,5],ui16>
    return %0 : !torch.vtensor<[3,4,5],ui16>
  }
}

