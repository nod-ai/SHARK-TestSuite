module {
  func.func @test_max_uint8(%arg0: !torch.vtensor<[3],ui8>, %arg1: !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Max"(%arg0, %arg1) : (!torch.vtensor<[3],ui8>, !torch.vtensor<[3],ui8>) -> !torch.vtensor<[3],ui8>
    return %0 : !torch.vtensor<[3],ui8>
  }
}

