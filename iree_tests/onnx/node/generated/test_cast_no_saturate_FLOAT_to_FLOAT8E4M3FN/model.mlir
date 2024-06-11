module {
  func.func public @test_cast_no_saturate_FLOAT_to_FLOAT8E4M3FN(%arg0: !torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,5],f8E4M3FN> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.saturate = 0 : si64, torch.onnx.to = 17 : si64} : (!torch.vtensor<[3,5],f32>) -> !torch.vtensor<[3,5],f8E4M3FN> 
    return %0 : !torch.vtensor<[3,5],f8E4M3FN>
  }
}

