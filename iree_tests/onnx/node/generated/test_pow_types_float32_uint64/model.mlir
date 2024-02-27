module {
  func.func @test_pow_types_float32_uint64(%arg0: !torch.vtensor<[3],f32>, %arg1: !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],f32> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 15 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Pow"(%arg0, %arg1) : (!torch.vtensor<[3],f32>, !torch.vtensor<[3],ui64>) -> !torch.vtensor<[3],f32>
    return %0 : !torch.vtensor<[3],f32>
  }
}

