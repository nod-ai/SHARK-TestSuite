module {
  func.func @test_bitwise_xor_i32_2d(%arg0: !torch.vtensor<[3,4],si32>, %arg1: !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.BitwiseXor"(%arg0, %arg1) : (!torch.vtensor<[3,4],si32>, !torch.vtensor<[3,4],si32>) -> !torch.vtensor<[3,4],si32>
    return %0 : !torch.vtensor<[3,4],si32>
  }
}

