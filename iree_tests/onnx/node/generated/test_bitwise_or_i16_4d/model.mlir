module {
  func.func @test_bitwise_or_i16_4d(%arg0: !torch.vtensor<[3,4,5,6],si8>, %arg1: !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.BitwiseOr"(%arg0, %arg1) : (!torch.vtensor<[3,4,5,6],si8>, !torch.vtensor<[3,4,5,6],si8>) -> !torch.vtensor<[3,4,5,6],si8>
    return %0 : !torch.vtensor<[3,4,5,6],si8>
  }
}

