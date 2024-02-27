module {
  func.func @test_bitwise_and_ui64_bcast_3v1d(%arg0: !torch.vtensor<[3,4,5],ui64>, %arg1: !torch.vtensor<[5],ui64>) -> !torch.vtensor<[3,4,5],ui64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.BitwiseAnd"(%arg0, %arg1) : (!torch.vtensor<[3,4,5],ui64>, !torch.vtensor<[5],ui64>) -> !torch.vtensor<[3,4,5],ui64>
    return %0 : !torch.vtensor<[3,4,5],ui64>
  }
}

