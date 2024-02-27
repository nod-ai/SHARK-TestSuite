module {
  func.func @test_mod_mixed_sign_int64(%arg0: !torch.vtensor<[6],si64>, %arg1: !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Mod"(%arg0, %arg1) : (!torch.vtensor<[6],si64>, !torch.vtensor<[6],si64>) -> !torch.vtensor<[6],si64>
    return %0 : !torch.vtensor<[6],si64>
  }
}

