module {
  func.func @test_cumsum_1d_reverse(%arg0: !torch.vtensor<[5],f64>, %arg1: !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 14 : si64, torch.onnx_meta.producer_name = "backend-test", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.CumSum"(%arg0, %arg1) {torch.onnx.reverse = 1 : si64} : (!torch.vtensor<[5],f64>, !torch.vtensor<[],si32>) -> !torch.vtensor<[5],f64>
    return %0 : !torch.vtensor<[5],f64>
  }
}

