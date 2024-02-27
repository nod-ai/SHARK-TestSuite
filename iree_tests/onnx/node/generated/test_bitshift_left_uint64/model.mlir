module {
  func.func @test_bitshift_left_uint64(%arg0: !torch.vtensor<[3],ui64>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 11 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.BitShift"(%arg0, %arg1) {torch.onnx.direction = "LEFT"} : (!torch.vtensor<[3],ui64>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],ui64>
    return %0 : !torch.vtensor<[3],ui64>
  }
}

