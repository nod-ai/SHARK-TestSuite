module {
  func.func @test_triu_square(%arg0: !torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Trilu"(%arg0) : (!torch.vtensor<[2,3,3],si64>) -> !torch.vtensor<[2,3,3],si64>
    return %0 : !torch.vtensor<[2,3,3],si64>
  }
}

